"""RFC 1982 序列号算术（32 位无符号整数环形空间）。

序列号空间为模 2^32 的环。给定当前值 ``current`` 与拟发布值 ``proposed``，
定义前向（顺时针）模距离 ``d = (proposed - current) mod 2^32``，则：

* ``d == 0``          -> ``equal``      （相等）
* ``1 <= d < 2^31``   -> ``newer``      （严格更新，可比较）
* ``d == 2^31``       -> ``ambiguous``  （正对点，RFC 1982 规定不可比较）
* ``2^31 < d < 2^32`` -> ``older``      （回退，可比较）

参考：RFC 1982 <https://www.rfc-editor.org/rfc/rfc1982>，SERIAL-BOUND = 2^31。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

BITS = 32
MOD = 1 << BITS          # 2^32，序列号空间大小
HALF = 1 << (BITS - 1)   # 2^31，SERIAL-BOUND
MAX_SERIAL = MOD - 1     # 2^32 - 1，合法序列号上界

NEWER = "newer"
EQUAL = "equal"
OLDER = "older"
AMBIGUOUS = "ambiguous"


def forward_distance(current: int, proposed: int) -> int:
    """返回 *proposed* 相对 *current* 的前向模距离，取值 [0, 2^32 - 1]。"""
    return (proposed - current) % MOD


def compare_serial(current: int, proposed: int) -> str:
    """按 RFC 1982 判定 *proposed* 相对 *current* 的关系。

    返回 ``"newer"`` / ``"equal"`` / ``"older"`` / ``"ambiguous"``。
    """
    distance = forward_distance(current, proposed)
    if distance == 0:
        return EQUAL
    if distance == HALF:
        return AMBIGUOUS
    if distance < HALF:
        return NEWER
    return OLDER


def is_strictly_newer(current: int, proposed: int) -> bool:
    """*proposed* 是否对 *current* 严格更新且可比较（1 <= 距离 <= 2^31 - 1）。"""
    return compare_serial(current, proposed) == NEWER


def propose_serial(currents: List[int]) -> Optional[Tuple[int, int]]:
    """推导对所有副本都严格更新且可比较的最佳拟发布值。

    对每个副本 ``c_i``，候选 ``x`` 必须满足
    ``1 <= (x - c_i) mod 2^32 <= 2^31 - 1``。

    优化目标：最小化最大前向距离 ``max_i (x - c_i) mod 2^32``；
    若多个候选同分，取数值最小者。

    可行域是每个副本所允许的开弧 ``{c+1, ..., c+2^31-1}``（模 2^32）
    之交，非空时必为一段连续弧，其最小元素必是某个 ``(c + 1) mod 2^32``；
    而沿弧每前进一步，目标函数恰好增加 1，故只需枚举这些下界候选。

    返回 ``(proposed, max_forward_distance)``；不存在可行候选时返回 ``None``。
    """
    unique_current = set(currents)
    candidates = [(c + 1) % MOD for c in unique_current]

    feasible: List[Tuple[int, int]] = []
    for candidate in candidates:
        distances = [forward_distance(c, candidate) for c in unique_current]
        if all(1 <= d <= HALF - 1 for d in distances):
            feasible.append((max(distances), candidate))

    if not feasible:
        return None

    # 先按最大前向距离排序，同分再按候选数值排序
    max_distance, proposed = min(feasible, key=lambda item: (item[0], item[1]))
    return proposed, max_distance


# ---------------------------------------------------------------------------
# IXFR 增量链（POST /transfer-plan）
# ---------------------------------------------------------------------------

UP_TO_DATE = "up_to_date"
IXFR = "ixfr"
AXFR = "axfr"

# axfr 的 reason
GAP = "gap"
"""增量链存在缺口（某序列号没有对应的下一条增量）。"""
LOOP = "loop"
"""沿增量链行进时回到了已经过的序列号，形成环路。"""
TARGET_INCOMPARABLE = "target_incomparable"
"""副本当前值与目标序列号在 RFC 1982 下不可比较（前向距离恰为 2^31）。"""
BUDGET = "budget"
"""完整增量集合下存在到目标的完整链，但受字节预算限制未被保留。"""

MAX_RETENTION_REPLICAS = 12
"""POST /retention-plan 与 /domain-retention-plan 允许的副本数上限
（选择算法枚举副本子集，复杂度 2^n）。"""


class DeltaValidationError(ValueError):
    """增量集合非法：存在非严格 newer 的增量、同一 from_serial 分叉，
    或同一增量携带冲突的字节数。"""

    def __init__(self, detail: dict):
        super().__init__(str(detail))
        self.detail = detail


def _check_strictly_newer(deltas: List[Tuple[int, ...]]) -> None:
    """逐条拒绝 to_serial 非严格 newer 的增量（保留输入顺序，报错确定可复现）。

    每条增量取前两个元素为 ``(from_serial, to_serial)``，其余元素（如字节数）
    不参与判定。
    """
    for delta in deltas:
        from_serial, to_serial = delta[0], delta[1]
        relation = compare_serial(from_serial, to_serial)
        if relation != NEWER:
            raise DeltaValidationError(
                {
                    "error": "delta_not_strictly_newer",
                    "message": (
                        "增量的 to_serial 必须按 RFC 1982 严格新于 from_serial"
                        "（前向距离在 [1, 2^31-1]）"
                    ),
                    "from_serial": from_serial,
                    "to_serial": to_serial,
                    "relation": relation,
                }
            )


def build_delta_chain(
    deltas: List[Tuple[int, int]],
) -> dict:
    """校验并归一化一组无序 IXFR 增量，返回 ``from_serial -> to_serial`` 映射。

    处理顺序：

    1. 每条增量都必须满足 ``to_serial`` 按 RFC 1982 **严格新于**
       ``from_serial``（前向距离在 ``[1, 2^31-1]``），否则抛出
       ``DeltaValidationError``（error=delta_not_strictly_newer）；
    2. 完全相同的 ``(from_serial, to_serial)`` 重复增量去重；
    3. 同一 ``from_serial`` 指向不同 ``to_serial`` 即链分叉，抛出
       ``DeltaValidationError``（error=delta_fork）。

    校验后每个 from_serial 至多有一个后继，因此任何完整链都是唯一的。
    """
    # 第一步：逐条拒绝非严格 newer 的增量
    _check_strictly_newer(deltas)

    # 第二步 + 第三步：重复去重，分叉拒绝
    chain: dict = {}
    for from_serial, to_serial in deltas:
        if from_serial in chain:
            if chain[from_serial] == to_serial:
                continue  # 重复增量，去重
            raise _fork_error(from_serial, chain[from_serial], to_serial)
        chain[from_serial] = to_serial
    return chain


def _fork_error(from_serial: int, existing: int, new: int) -> DeltaValidationError:
    return DeltaValidationError(
        {
            "error": "delta_fork",
            "message": (
                f"同一 from_serial={from_serial} 指向不同的 to_serial，"
                "增量链不允许分叉"
            ),
            "from_serial": from_serial,
            "to_serials": [existing, new],
        }
    )


def build_sized_delta_chain(
    deltas: List[Tuple[int, int, int]],
) -> Tuple[dict, dict]:
    """校验并归一化一组带字节数的无序 IXFR 增量。

    返回 ``(chain, sizes)``：``chain`` 为 ``from_serial -> to_serial`` 映射，
    ``sizes`` 为 ``(from_serial, to_serial) -> bytes`` 映射。

    处理顺序（与 :func:`build_delta_chain` 一致，先验合法性再去重/查冲突）：

    1. 每条增量都必须严格 newer，否则抛出 ``DeltaValidationError``
       （error=delta_not_strictly_newer）；
    2. 完全相同的 ``(from_serial, to_serial, bytes)`` 重复增量去重；
    3. 按输入顺序逐条判定冲突：同一 ``(from_serial, to_serial)`` 携带不同
       bytes 即字节数冲突，抛出 ``DeltaValidationError``
       （error=delta_bytes_conflict）；同一 ``from_serial`` 指向不同
       ``to_serial`` 即链分叉，抛出 ``DeltaValidationError``
       （error=delta_fork）。
    """
    # 第一步：逐条拒绝非严格 newer 的增量
    _check_strictly_newer(deltas)

    # 第二步 + 第三步：重复去重，字节数冲突与分叉拒绝
    chain: dict = {}
    sizes: dict = {}
    for from_serial, to_serial, size in deltas:
        if from_serial in chain:
            if chain[from_serial] != to_serial:
                raise _fork_error(from_serial, chain[from_serial], to_serial)
            if sizes[(from_serial, to_serial)] != size:
                raise DeltaValidationError(
                    {
                        "error": "delta_bytes_conflict",
                        "message": (
                            f"增量 ({from_serial}, {to_serial}) 携带了"
                            "不同的字节数，相同增量的 bytes 必须一致"
                        ),
                        "from_serial": from_serial,
                        "to_serial": to_serial,
                        "bytes": [sizes[(from_serial, to_serial)], size],
                    }
                )
            continue  # 完全相同的重复增量，去重
        chain[from_serial] = to_serial
        sizes[(from_serial, to_serial)] = size
    return chain, sizes


def plan_transfer(
    current: int,
    target: int,
    chain: dict,
) -> Tuple[str, Optional[str], List[Tuple[int, int]]]:
    """为单个副本规划从 *current* 到 *target* 的传输方式。

    *chain* 为 :func:`build_delta_chain` 产出的 ``from_serial -> to_serial``
    映射，每条边都是 RFC 1982 严格推进。沿边逐跳追踪（**不做整数排序**，
    仅以增量自带的环形先后关系为准）：

    * ``current == target``：``(up_to_date, None, [])``；
    * 两端在 RFC 1982 下不可比较（前向距离恰为 2^31）：
      ``(axfr, target_incomparable, [])``，即使存在能连通的边也必须 AXFR；
    * 沿唯一增量链可抵达 *target*：``(ixfr, None, ordered_deltas)``；
    * 某一跳找不到后继：``(axfr, gap, [])``；
    * 行进中重复经过某序列号（环路）：``(axfr, loop, [])``。
    """
    if current == target:
        return UP_TO_DATE, None, []

    if compare_serial(current, target) == AMBIGUOUS:
        return AXFR, TARGET_INCOMPARABLE, []

    ordered: List[Tuple[int, int]] = []
    visited = {current}
    node = current
    while node != target:
        nxt = chain.get(node)
        if nxt is None:
            return AXFR, GAP, []
        if nxt in visited:
            return AXFR, LOOP, []
        ordered.append((node, nxt))
        visited.add(nxt)
        node = nxt
    return IXFR, None, ordered


# ---------------------------------------------------------------------------
# 增量保留规划（POST /retention-plan、POST /domain-retention-plan）
# ---------------------------------------------------------------------------


def _requirement_masks(
    target: int,
    replicas: List[Tuple[int, int]],
    chain: dict,
    edge_id: dict,
) -> Tuple[List[Optional[int]], List[Optional[str]]]:
    """在**完整**增量集合上为每个副本求需求集（增量位掩码）与固有原因。

    依次跑 :func:`plan_transfer`：

    * ``ixfr``：链上增量编号取位 OR，得到该副本的需求集掩码；
    * ``up_to_date``：需求集为 0（恒被覆盖，零成本），固有原因为 None；
    * ``axfr``（``gap`` / ``loop`` / ``target_incomparable``）：掩码为 None，
      保留其固有原因（任何保留子集都无法改变）。
    """
    req_masks: List[Optional[int]] = []
    inherent_reasons: List[Optional[str]] = []
    for current, _priority in replicas:
        mode, reason, ordered = plan_transfer(current, target, chain)
        if mode == IXFR:
            mask = 0
            for edge in ordered:
                mask |= 1 << edge_id[edge]
            req_masks.append(mask)
            inherent_reasons.append(None)
        elif mode == UP_TO_DATE:
            req_masks.append(0)
            inherent_reasons.append(None)
        else:
            req_masks.append(None)
            inherent_reasons.append(reason)
    return req_masks, inherent_reasons


def _mask_bytes(mask: int, edge_bytes: List[int]) -> int:
    """位掩码中选中增量的字节总数。"""
    used = 0
    bits = mask
    while bits:
        bit = bits & (-bits)
        used += edge_bytes[bit.bit_length() - 1]
        bits ^= bit
    return used


def _mask_sequence(
    mask: int, edges: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """位掩码展开为按 ``(from_serial, to_serial)`` 升序的增量序列。"""
    sequence: List[Tuple[int, int]] = []
    bits = mask
    while bits:
        bit = bits & (-bits)
        sequence.append(edges[bit.bit_length() - 1])
        bits ^= bit
    return sequence


def plan_retention(
    target: int,
    replicas: List[Tuple[int, int]],
    chain: dict,
    sizes: dict,
    budget: int,
) -> Tuple[List[Tuple[int, int]], int, List[Tuple[str, Optional[str]]]]:
    """在字节预算内选择要保留的增量子集。

    *replicas* 为 ``(current, priority)`` 二元组（至多
    ``MAX_RETENTION_REPLICAS`` 个）；*chain* / *sizes* 为
    :func:`build_sized_delta_chain` 的产物；*budget* 为非负总字节预算。

    先用 :func:`plan_transfer` 在**完整**增量集合上判定每个副本：

    * ``up_to_date``：当前值已等于目标，无需任何增量，恒视为已覆盖；
    * ``ixfr``：存在唯一完整链，链上增量即该副本的**需求集**——需求集
      被保留子集包含时标记 ``ixfr``，否则 ``axfr / budget``；
    * ``axfr``（``gap`` / ``loop`` / ``target_incomparable``）：完整集合
      下即不可达，保留任何子集都无法改变，原因保持。

    在所有满足 ``字节总数 <= budget`` 的保留子集中，依次：

    1. 最大化被覆盖副本（拥有到目标完整链）的 priority 总和；
    2. 最大化被覆盖副本数；
    3. 最小化占用字节数；
    4. 取按 ``(from_serial, to_serial)`` 升序排列的增量序列中字典序最小者。

    保留集必为被覆盖副本需求集之并（多保留对覆盖无贡献的增量只会浪费
    预算），因此枚举全部副本子集（至多 2^12）即可求得精确最优；
    预算无法保留任何完整链时，最优即空集（仍正常返回）。

    返回 ``(retained, used_bytes, results)``：``retained`` 为按
    ``(from_serial, to_serial)`` 升序的增量列表；``results`` 与输入副本
    同序，每项为 ``(mode, reason)``。
    """
    # 增量按 (from_serial, to_serial) 升序编号：按位展开即为有序序列
    edges = sorted(sizes)
    edge_id = {edge: i for i, edge in enumerate(edges)}
    edge_bytes = [sizes[edge] for edge in edges]

    # 第一遍：完整集合上判定，求出每个副本的需求集（增量位掩码）
    # None 表示完整集合下即不可达（gap / loop / target_incomparable）
    req_masks, inherent_reasons = _requirement_masks(
        target, replicas, chain, edge_id
    )

    # up_to_date 副本恒被覆盖，作为常量计入目标函数
    base_prio = sum(
        prio for (_, prio), mask in zip(replicas, req_masks) if mask == 0
    )
    base_count = sum(1 for mask in req_masks if mask == 0)

    # 只需枚举"完整集合下为 ixfr"的副本子集（至多 12 个 -> 至多 4096 个子集）
    enum_idx = [i for i, mask in enumerate(req_masks) if mask]
    reqs = [req_masks[i] for i in enum_idx]
    prios = [replicas[i][1] for i in enum_idx]
    k = len(enum_idx)

    union_of = [0] * (1 << k)
    best_key: Optional[Tuple[int, int, int]] = None
    best_unions: List[int] = []
    for subset in range(1 << k):
        lsb = subset & (-subset)
        if lsb:
            union = union_of[subset ^ lsb] | reqs[lsb.bit_length() - 1]
        else:
            union = 0
        union_of[subset] = union

        used = _mask_bytes(union, edge_bytes)
        if used > budget:
            continue

        prio = base_prio
        count = base_count
        for req, p in zip(reqs, prios):
            if req & ~union == 0:  # 需求集被保留集包含即被覆盖
                prio += p
                count += 1

        key = (prio, count, -used)
        if best_key is None or key > best_key:
            best_key = key
            best_unions = [union]
        elif key == best_key:
            best_unions.append(union)

    # 最后一级：增量序列（按 (from_serial, to_serial) 升序）字典序最小
    best_union = min(best_unions, key=lambda union: _mask_sequence(union, edges))
    retained = _mask_sequence(best_union, edges)

    # 空子集恒可行（budget >= 0），best_unions 必非空
    used_bytes = sum(sizes[edge] for edge in retained)

    results: List[Tuple[str, Optional[str]]] = []
    for mask, reason in zip(req_masks, inherent_reasons):
        if mask is None:
            results.append((AXFR, reason))
        elif mask == 0:
            results.append((UP_TO_DATE, None))
        elif mask & ~best_union == 0:
            results.append((IXFR, None))
        else:
            results.append((AXFR, BUDGET))
    return retained, used_bytes, results


def plan_domain_retention(
    target: int,
    replicas: List[Tuple[int, int, str]],
    chain: dict,
    sizes: dict,
    budget: int,
) -> Tuple[
    List[Tuple[int, int]],
    int,
    List[Tuple[str, Optional[str]]],
    List[Tuple[str, int, int]],
]:
    """在字节预算内按 failure_domain 维度选择要保留的增量子集。

    *replicas* 为 ``(current, priority, failure_domain)`` 三元组（至多
    ``MAX_RETENTION_REPLICAS`` 个，``failure_domain`` 非空）；*chain* /
    *sizes* 为 :func:`build_sized_delta_chain` 的产物；*budget* 为非负总字节
    预算。每个副本在完整集合下的可达性判定与 :func:`plan_retention` 完全一致
    （复用 :func:`plan_transfer`，``gap`` / ``loop`` /
    ``target_incomparable`` 原因保持）。

    对任一保留子集，先按 failure_domain 汇总该域已覆盖（``ixfr`` 与
    ``up_to_date``）副本的 priority 总和，再把各域总和按**升序**排列成向量。
    在所有满足 ``字节总数 <= budget`` 的保留子集中，依次：

    1. 各域已覆盖 priority 总和的**升序向量字典序最大**（逐域 maximin）；
    2. 全体已覆盖副本的 **priority 总和最大**；
    3. 已覆盖**副本数最多**；
    4. **占用字节最少**；
    5. 取按 ``(from_serial, to_serial)`` 升序排列的增量序列中字典序最小者。

    向量含输入中出现的**全部** failure_domain（按名称升序排列）：该域无任何
    已覆盖副本时对应分量为 0，从而域的排列不依赖于哪个子集被选中。

    与 :func:`plan_retention` 相同，保留集必为被覆盖 ixfr 副本需求集之并，
    枚举全部 ixfr 副本子集（至多 2^12）即可精确求解；空子集恒可行。

    返回 ``(retained, used_bytes, results, domain_summaries)``：``retained``
    为按 ``(from_serial, to_serial)`` 升序的增量列表；``results`` 与输入副本
    同序，每项为 ``(mode, reason)``；``domain_summaries`` 为按 failure_domain
    名称升序的 ``(failure_domain, covered_priority, total_priority)`` 列表，
    其中 ``total_priority`` 为该域**全部**副本的 priority 之和（不随子集变化）。
    """
    # 增量按 (from_serial, to_serial) 升序编号：按位展开即为有序序列
    edges = sorted(sizes)
    edge_id = {edge: i for i, edge in enumerate(edges)}
    edge_bytes = [sizes[edge] for edge in edges]

    # failure_domain 按名称升序固定编号，向量分量顺序与子集无关
    domains = sorted({domain for _c, _p, domain in replicas})
    domain_id = {domain: i for i, domain in enumerate(domains)}
    replica_domain = [domain_id[domain] for _c, _p, domain in replicas]
    domain_total = [0] * len(domains)
    for (_c, priority, _domain), di in zip(replicas, replica_domain):
        domain_total[di] += priority

    # 完整集合上判定，求出每个副本的需求集（增量位掩码）与固有原因
    replica_base = [(current, priority) for current, priority, _d in replicas]
    req_masks, inherent_reasons = _requirement_masks(
        target, replica_base, chain, edge_id
    )

    # up_to_date 副本恒被覆盖（零成本），其 priority 作为常量并入域向量
    base_vector = [0] * len(domains)
    base_prio = 0
    base_count = 0
    for (mask, (_c, priority, _d), di) in zip(
        req_masks, replicas, replica_domain
    ):
        if mask == 0:
            base_vector[di] += priority
            base_prio += priority
            base_count += 1

    # 只需枚举"完整集合下为 ixfr"的副本子集（至多 12 个 -> 至多 4096 个子集）
    enum_idx = [i for i, mask in enumerate(req_masks) if mask]
    reqs = [req_masks[i] for i in enum_idx]
    prios = [replicas[i][1] for i in enum_idx]
    req_domains = [replica_domain[i] for i in enum_idx]
    k = len(enum_idx)

    union_of = [0] * (1 << k)
    # key = (域向量, 全体 priority, 覆盖数, -字节数)；域向量升序字典序最大化。
    # 域向量须为 tuple（list 之间不支持 < / >，无法参与字典序比较）。
    best_key: Optional[Tuple[Tuple[int, ...], int, int, int]] = None
    best_unions: List[int] = []
    for subset in range(1 << k):
        lsb = subset & (-subset)
        if lsb:
            union = union_of[subset ^ lsb] | reqs[lsb.bit_length() - 1]
        else:
            union = 0
        union_of[subset] = union

        used = _mask_bytes(union, edge_bytes)
        if used > budget:
            continue

        covered = [0] * len(domains)
        prio = base_prio
        count = base_count
        for req, p, di in zip(reqs, prios, req_domains):
            if req & ~union == 0:  # 需求集被保留集包含即被覆盖
                covered[di] += p
                prio += p
                count += 1

        vector = tuple(
            sorted(base_vector[i] + covered[i] for i in range(len(domains)))
        )
        key = (vector, prio, count, -used)
        if best_key is None or key > best_key:
            best_key = key
            best_unions = [union]
        elif key == best_key:
            best_unions.append(union)

    # 最后一级：增量序列（按 (from_serial, to_serial) 升序）字典序最小
    best_union = min(best_unions, key=lambda union: _mask_sequence(union, edges))
    retained = _mask_sequence(best_union, edges)

    # 空子集恒可行（budget >= 0），best_unions 必非空
    used_bytes = sum(sizes[edge] for edge in retained)

    results: List[Tuple[str, Optional[str]]] = []
    domain_covered = [0] * len(domains)
    for mask, reason, (_, priority, _d), di in zip(
        req_masks, inherent_reasons, replicas, replica_domain
    ):
        if mask is None:
            results.append((AXFR, reason))
        elif mask == 0:
            results.append((UP_TO_DATE, None))
            domain_covered[di] += priority
        elif mask & ~best_union == 0:
            results.append((IXFR, None))
            domain_covered[di] += priority
        else:
            results.append((AXFR, BUDGET))

    domain_summaries = [
        (domain, domain_covered[domain_id[domain]], total)
        for domain, total in zip(domains, domain_total)
    ]
    return retained, used_bytes, results, domain_summaries


# ---------------------------------------------------------------------------
# 上线门禁（POST /rollout-gate）
# ---------------------------------------------------------------------------

MISSING_TIME_ORDER = "missing_time_order"
"""相邻观测至少一条缺失时间戳，无法建立时间顺序（不判定序列号关系）。"""

BASELINE_NOT_REACHED = "baseline_not_reached"
"""副本的观测序列中没有任何一条达到基线序列号。"""

QUORUM_EXCEEDS_REPLICAS = "quorum_exceeds_replicas"
"""所需 quorum 超过副本总数，任何时刻都不可能满足。"""

QUORUM_NEVER_REACHED = "quorum_never_reached"
"""峰值同时处于有效推进区间内的副本数仍低于 quorum。"""


class ObservationOrderError(ValueError):
    """同一副本的观测时间戳重复或倒序。"""

    def __init__(self, detail: dict):
        super().__init__(str(detail))
        self.detail = detail


def check_observation_time_order(
    name: str, timestamps: List[Optional[int]]
) -> None:
    """校验单个副本的观测时间戳严格递增（None 表示缺失，跳过不参与校验）。

    重复（与前一存在值相等）抛出 error=duplicate_observation_timestamp；
    倒序（小于前一存在值）抛出 error=observation_timestamp_out_of_order。
    两者均为 :class:`ObservationOrderError`，由路由层转为 422。
    """
    prev_index = -1
    prev: Optional[int] = None
    for index, ts in enumerate(timestamps):
        if ts is None:
            continue
        if prev is not None:
            if ts == prev:
                raise ObservationOrderError(
                    {
                        "error": "duplicate_observation_timestamp",
                        "message": (
                            f"副本 {name} 的观测时间戳重复：{ts}"
                            f"（下标 {prev_index} 与 {index}），时间戳必须严格递增"
                        ),
                        "replica": name,
                        "index": index,
                        "timestamp": ts,
                    }
                )
            if ts < prev:
                raise ObservationOrderError(
                    {
                        "error": "observation_timestamp_out_of_order",
                        "message": (
                            f"副本 {name} 的观测时间戳必须严格递增："
                            f"下标 {prev_index} 为 {prev}，下标 {index} 为 {ts}"
                        ),
                        "replica": name,
                        "index": index,
                        "timestamp": ts,
                        "previous_timestamp": prev,
                    }
                )
        prev = ts
        prev_index = index


def reaches_serial(baseline: int, serial: int) -> bool:
    """*serial* 是否达到 *baseline*（相等，或按 RFC 1982 严格更新）。

    即 ``(serial - baseline) mod 2^32 ∈ [0, 2^31 - 1]``；相对基线为
    ``older`` 或 ``ambiguous``（正对点）的序列号不算达到。
    """
    return compare_serial(baseline, serial) in (EQUAL, NEWER)


def evaluate_replica_rollout(
    baseline: int,
    observations: List[Tuple[Optional[int], int]],
) -> Tuple[
    bool,
    Optional[int],
    Optional[Tuple[int, int]],
    Optional[str],
    List[Tuple[int, str]],
]:
    """评估单个副本的观测序列相对基线的推进情况。

    *observations* 为按时间排序的 ``(timestamp, serial)`` 序列（timestamp 可
    缺失；存在的时间戳必须严格递增，由 :func:`check_observation_time_order`
    事先校验）。

    相邻观测对的关系标记：

    * 任一时间戳缺失：``missing_time_order``（无法建立时间顺序，不判定序列号
      关系）；
    * 否则按 RFC 1982 判定：``newer``（严格推进，允许跨 2^32 回绕）/
      ``equal`` / ``older`` / ``ambiguous``。

    首次达到基线 = 首个满足 :func:`reaches_serial` 的观测。**有效推进区间**
    从该观测起沿相邻对逐步扩展，每一步都必须同时满足：

    1. 时间戳齐全（非 ``missing_time_order``）；
    2. 关系为 ``newer`` 或 ``equal``（``older`` / ``ambiguous`` 为回退或
       不可比较，终止区间）；
    3. 新观测仍达到基线——一次 ``newer`` 推进可能跨越大半环后重新跌破基线
       （如基线 10：``10 -> 10+2^31-1 -> 8``，第二步仍是 newer 但 8 未达到
       基线），故每一步都重新校验。

    区间为闭区间 ``[首次达到的时间戳, 最后一个有效观测的时间戳]``；
    区间之外副本状态未知，不计入事件扫掠。

    返回 ``(reaches, first_reached_at, interval, reason, transitions)``：

    * ``reaches``：是否有任一观测达到基线；
    * ``first_reached_at``：首次达到基线的观测时间戳（该观测无时间戳时为
      None）；
    * ``interval``：``(start, end)`` 闭区间，无区间时为 None；
    * ``reason``：无区间时的原因——``baseline_not_reached``（从未达到）或
      ``missing_time_order``（达到了但首次达到的观测无时间戳，无法定位区间）；
    * ``transitions``：``(后一条观测下标, 关系标记)`` 列表，按输入顺序。
    """
    transitions: List[Tuple[int, str]] = []
    for i in range(1, len(observations)):
        t_prev, _s_prev = observations[i - 1]
        t_curr, _s_curr = observations[i]
        if t_prev is None or t_curr is None:
            relation = MISSING_TIME_ORDER
        else:
            relation = compare_serial(observations[i - 1][1], observations[i][1])
        transitions.append((i, relation))

    first: Optional[int] = None
    for i, (_ts, serial) in enumerate(observations):
        if reaches_serial(baseline, serial):
            first = i
            break
    if first is None:
        return False, None, None, BASELINE_NOT_REACHED, transitions

    first_reached_at = observations[first][0]
    if first_reached_at is None:
        return True, None, None, MISSING_TIME_ORDER, transitions

    end = first
    for i in range(first, len(observations) - 1):
        relation = transitions[i][1]  # transitions[i] 对应相邻对 (i, i+1)
        if relation not in (NEWER, EQUAL):
            break
        if not reaches_serial(baseline, observations[i + 1][1]):
            break
        end = i + 1
    interval = (first_reached_at, observations[end][0])
    return True, first_reached_at, interval, None, transitions


def sweep_quorum(
    intervals: List[Tuple[int, int]], quorum: int
) -> Tuple[Optional[int], Optional[int], int]:
    """事件扫掠：计算 quorum 首次满足与首次满足后再次跌破的时间。

    *intervals* 为各副本的有效推进**闭区间** ``[start, end]``（端点处副本仍
    计入）。并发数只在区间端点处变化，因此只在事件时间点上评估。

    返回 ``(first_satisfied_at, dropped_below_at, peak)``：

    * ``first_satisfied_at``：并发数首次 >= quorum 的事件时间戳；从未满足为
      None；
    * ``dropped_below_at``：首次满足之后并发数首次 < quorum 的事件时间戳；
      满足后直到观测结束未再跌破为 None；
    * ``peak``：所有事件时间点上的峰值并发数（无区间为 0）。
    """
    starts: dict = {}
    ends: dict = {}
    for start, end in intervals:
        starts[start] = starts.get(start, 0) + 1
        ends[end] = ends.get(end, 0) + 1

    first_satisfied_at: Optional[int] = None
    dropped_below_at: Optional[int] = None
    peak = 0
    active = 0
    for t in sorted(set(starts) | set(ends)):
        active += starts.get(t, 0)
        # 闭区间：end == t 的副本在 t 时刻仍计入，先计数再移除
        if active > peak:
            peak = active
        if first_satisfied_at is None:
            if active >= quorum:
                first_satisfied_at = t
        elif dropped_below_at is None and active < quorum:
            dropped_below_at = t
        active -= ends.get(t, 0)
    return first_satisfied_at, dropped_below_at, peak


def evaluate_rollout_gate(
    baseline: int,
    quorum: int,
    replica_observations: List[List[Tuple[Optional[int], int]]],
) -> Tuple[list, bool, Optional[int], Optional[int], int, Optional[str]]:
    """对全部副本做推进评估，并用事件扫掠判定 quorum 时间窗。

    *replica_observations* 与输入副本同序，每项为该副本按时间排序的
    ``(timestamp, serial)`` 观测序列（同 :func:`evaluate_replica_rollout`）。

    返回 ``(results, satisfied, first_satisfied_at, dropped_below_at, peak,
    reason)``：``results`` 与输入副本同序，每项为
    :func:`evaluate_replica_rollout` 的返回元组；``reason`` 在 quorum 从未
    满足时给出明确原因——``quorum_exceeds_replicas``（quorum 超过副本总数）
    或 ``quorum_never_reached``（峰值并发不足），满足时为 None。
    """
    results = []
    intervals: List[Tuple[int, int]] = []
    for observations in replica_observations:
        outcome = evaluate_replica_rollout(baseline, observations)
        results.append(outcome)
        interval = outcome[2]
        if interval is not None:
            intervals.append(interval)

    first_satisfied_at, dropped_below_at, peak = sweep_quorum(intervals, quorum)
    satisfied = first_satisfied_at is not None
    if satisfied:
        reason = None
    elif quorum > len(replica_observations):
        reason = QUORUM_EXCEEDS_REPLICAS
    else:
        reason = QUORUM_NEVER_REACHED
    return results, satisfied, first_satisfied_at, dropped_below_at, peak, reason


# ---------------------------------------------------------------------------
# 传播 SLO（POST /propagation-slo）
# ---------------------------------------------------------------------------

DUPLICATE_RELEASE_TIMESTAMP = "duplicate_release_timestamp"
"""相邻两次发布事件使用了相同时间戳（必须严格递增）。"""

RELEASE_TIMESTAMP_OUT_OF_ORDER = "release_timestamp_out_of_order"
"""发布事件时间戳倒序（必须严格递增）。"""


class ReleaseOrderError(ValueError):
    """发布事件的时间戳重复或未严格递增。"""

    def __init__(self, detail: dict):
        super().__init__(str(detail))
        self.detail = detail


def check_release_time_order(timestamps: List[int]) -> None:
    """校验发布事件时间戳严格递增。

    与前一时间戳相等抛出 error=duplicate_release_timestamp；小于前一时间戳
    抛出 error=release_timestamp_out_of_order。两者均为
    :class:`ReleaseOrderError`，由路由层转为 422。
    """
    prev: Optional[int] = None
    prev_index = -1
    for index, ts in enumerate(timestamps):
        if prev is not None:
            if ts == prev:
                raise ReleaseOrderError(
                    {
                        "error": DUPLICATE_RELEASE_TIMESTAMP,
                        "message": (
                            f"发布时间戳重复：{ts}（下标 {prev_index} 与 {index}），"
                            "发布事件必须按时间严格递增"
                        ),
                        "index": index,
                        "timestamp": ts,
                    }
                )
            if ts < prev:
                raise ReleaseOrderError(
                    {
                        "error": RELEASE_TIMESTAMP_OUT_OF_ORDER,
                        "message": (
                            "发布事件必须按时间严格递增："
                            f"下标 {prev_index} 为 {prev}，下标 {index} 为 {ts}"
                        ),
                        "index": index,
                        "timestamp": ts,
                        "previous_timestamp": prev,
                    }
                )
        prev = ts
        prev_index = index


def evaluate_propagation_slo(
    releases: List[Tuple[int, int]],
    failure_domains: List[str],
    replica_observations: List[List[Tuple[int, int]]],
    deadline_seconds: int,
) -> List[dict]:
    """逐次发布评估按故障域的传播收敛时延（SLO）。

    *releases* 为按时间严格递增的 ``(timestamp, serial)`` 发布事件（由
    :func:`check_release_time_order` 事先校验）；*failure_domains* 与
    *replica_observations* 均按副本顺序给出，后者每项为该副本的
    ``(timestamp, serial)`` 观测序列（同副本时间戳严格递增，由
    :func:`check_observation_time_order` 事先校验，跨副本时间戳可相同）；
    *deadline_seconds* 为非负 SLO 时限。

    对第 i 次发布只考察其**专属窗口** ``[released_at, 下一次发布时间)``
    （最后一次发布的窗口无上界）：窗口外（早于本次发布，或已进入下一次
    发布窗口）的观测一律不参与；每个窗口内副本的到达状态从零开始计算。

    窗口内把各副本观测合并为 ``(timestamp, 副本)`` 事件并按
    ``(时间戳, 副本下标)`` 确定地排序、逐时间戳成批处理。按 RFC 1982
    判定观测序列号是否**达到**发布版本（:func:`reaches_serial`：equal 或
    newer；older 与正对点 ambiguous 均不算达到，允许跨 2^32 回绕的 newer）：

    * 某副本首次给出达到版本的观测，即标记该副本已到达，并把其故障域的
      **首个有效到达时间**记为该观测时间戳（域内多副本取最早者）；
    * 已到达副本随后给出 older / ambiguous 观测，则该故障域此前的有效
      到达**失效**（首个有效到达时间清空、相关副本标记为未到达），须等待
      后续（域内任一副本的）达到观测重新建立——重新达到的时间重新记为
      首个有效到达时间；
    * 同一时间戳的事件作一个原子批：先统一处理失到达，再统一处理达到，
      故同批内失效率与重新达到可以相互抵消（结果不依赖同时间戳事件的
      排列方式）。

    每个时间戳批处理完后，若全部故障域都持有有效的首个到达时间即**收敛**：
    ``converged_at`` = 各域首个有效到达时间中的**最晚值**，此后该窗口冻结
    （收敛之后再出现 older / ambiguous 不改变结果）。窗口结束仍有域未
    覆盖则 ``converged_at=None``。

    每次发布返回一个 dict：

    * ``released_at`` / ``serial``：回显发布时间与版本；
    * ``converged_at``：收敛时间戳，未收敛为 None；
    * ``latency_seconds``：``converged_at - released_at``，未收敛为 None；
    * ``met_deadline``：收敛且 ``latency_seconds <= deadline_seconds`` 为
      True，否则 False；
    * ``domain_reached_at``：按故障域名升序的 ``(failure_domain, reached_at)``
      列表（未收敛时该域当前无有效到达则 reached_at=None；收敛后为收敛
      时刻的快照）；
    * ``missing_domains``：窗口结束仍无有效到达的故障域名（升序），收敛
      时为空列表。
    """
    domains = sorted(set(failure_domains))
    results: List[dict] = []

    for release_index, (released_at, serial) in enumerate(releases):
        window_end = (
            releases[release_index + 1][0]
            if release_index + 1 < len(releases)
            else None
        )

        # 收集本窗口 [released_at, window_end) 内的全部观测事件
        events: List[Tuple[int, int, int]] = []
        for replica_index, observations in enumerate(replica_observations):
            for ts, observed_serial in observations:
                if ts < released_at:
                    continue
                if window_end is not None and ts >= window_end:
                    continue
                events.append((ts, replica_index, observed_serial))
        events.sort(key=lambda event: (event[0], event[1]))

        replica_arrived = [False] * len(failure_domains)
        domain_reached: dict = {domain: None for domain in domains}
        converged_at: Optional[int] = None

        cursor = 0
        while cursor < len(events):
            ts = events[cursor][0]
            batch = []
            while cursor < len(events) and events[cursor][0] == ts:
                batch.append(events[cursor])
                cursor += 1

            # 第一步：已到达副本出现 older / ambiguous -> 其域到达失效。
            # 必须在更新副本状态之前用旧状态判定，且整批先统一失效，
            # 使同一时间戳的事件顺序不影响结果。
            for _t, replica_index, observed_serial in batch:
                if replica_arrived[replica_index] and not reaches_serial(
                    serial, observed_serial
                ):
                    domain_reached[failure_domains[replica_index]] = None

            # 第二步：统一处理达到 / 未达到，重新建立域的首个有效到达时间
            for _t, replica_index, observed_serial in batch:
                domain = failure_domains[replica_index]
                if reaches_serial(serial, observed_serial):
                    replica_arrived[replica_index] = True
                    if domain_reached[domain] is None:
                        domain_reached[domain] = ts
                else:
                    replica_arrived[replica_index] = False

            if converged_at is None and all(
                domain_reached[domain] is not None for domain in domains
            ):
                converged_at = max(
                    domain_reached[domain] for domain in domains
                )
                break  # 收敛后冻结：后续失到达不再影响本次发布结果

        if converged_at is None:
            latency: Optional[int] = None
            met_deadline = False
            missing_domains = [
                domain for domain in domains if domain_reached[domain] is None
            ]
        else:
            latency = converged_at - released_at
            met_deadline = latency <= deadline_seconds
            missing_domains = []

        results.append(
            {
                "released_at": released_at,
                "serial": serial,
                "converged_at": converged_at,
                "latency_seconds": latency,
                "met_deadline": met_deadline,
                "domain_reached_at": [
                    (domain, domain_reached[domain]) for domain in domains
                ],
                "missing_domains": missing_domains,
            }
        )

    return results
