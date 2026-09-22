"""POST /propagation-slo 传播 SLO 单元测试（仅依赖标准库 + 本项目模块）。

运行：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

from app.serial import (
    HALF,
    MAX_SERIAL,
    MOD,
    DUPLICATE_RELEASE_TIMESTAMP,
    RELEASE_TIMESTAMP_OUT_OF_ORDER,
    ReleaseOrderError,
    check_release_time_order,
    evaluate_propagation_slo,
)


class CheckReleaseTimeOrderTests(unittest.TestCase):
    def test_strictly_increasing_ok(self):
        check_release_time_order([])
        check_release_time_order([1])
        check_release_time_order([0, 1, 100, 1000])

    def test_duplicate_timestamp_rejected(self):
        with self.assertRaises(ReleaseOrderError) as ctx:
            check_release_time_order([10, 20, 20])
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], DUPLICATE_RELEASE_TIMESTAMP)
        self.assertEqual(detail["index"], 2)
        self.assertEqual(detail["timestamp"], 20)

    def test_out_of_order_timestamp_rejected(self):
        with self.assertRaises(ReleaseOrderError) as ctx:
            check_release_time_order([10, 30, 25])
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], RELEASE_TIMESTAMP_OUT_OF_ORDER)
        self.assertEqual(detail["index"], 2)
        self.assertEqual(detail["timestamp"], 25)
        self.assertEqual(detail["previous_timestamp"], 30)


class EvaluatePropagationSloTests(unittest.TestCase):
    def test_two_releases_two_domains(self):
        results = evaluate_propagation_slo(
            [(1000, 100), (2000, 101)],
            ["dc-east", "dc-west"],
            [
                [(1010, 100), (2005, 101)],
                [(1040, 100), (2060, 101)],
            ],
            60,
        )
        first, second = results
        self.assertEqual(first["released_at"], 1000)
        self.assertEqual(first["serial"], 100)
        self.assertEqual(first["converged_at"], 1040)
        self.assertEqual(first["latency_seconds"], 40)
        self.assertTrue(first["met_deadline"])
        self.assertEqual(
            first["domain_reached_at"], [("dc-east", 1010), ("dc-west", 1040)]
        )
        self.assertEqual(first["missing_domains"], [])

        self.assertEqual(second["converged_at"], 2060)
        self.assertEqual(second["latency_seconds"], 60)
        self.assertTrue(second["met_deadline"])  # latency == deadline 也算达标
        self.assertEqual(
            second["domain_reached_at"], [("dc-east", 2005), ("dc-west", 2060)]
        )

    def test_deadline_miss(self):
        (item,) = evaluate_propagation_slo(
            [(1000, 100)], ["a"], [[(1011, 100)]], 10
        )
        self.assertEqual(item["converged_at"], 1011)
        self.assertEqual(item["latency_seconds"], 11)
        self.assertFalse(item["met_deadline"])
        self.assertEqual(item["missing_domains"], [])

    def test_zero_latency_at_release_timestamp(self):
        # 发布时间戳同刻的达到观测即收敛：latency 0，deadline 0 也达标
        (item,) = evaluate_propagation_slo(
            [(1000, 100)], ["a"], [[(1000, 100)]], 0
        )
        self.assertEqual(item["converged_at"], 1000)
        self.assertEqual(item["latency_seconds"], 0)
        self.assertTrue(item["met_deadline"])

    def test_equal_and_newer_both_reach(self):
        results = evaluate_propagation_slo(
            [(1000, 100)],
            ["a", "b"],
            [[(1001, 100)], [(1002, 150)]],  # equal 与严格 newer
            60,
        )
        self.assertEqual(results[0]["converged_at"], 1002)

    def test_wrap_around_newer_reaches(self):
        # 4294967295 -> 0 是跨 2^32 回绕的合法 newer
        (item,) = evaluate_propagation_slo(
            [(1000, MAX_SERIAL)], ["a"], [[(1000, 0)]], 5
        )
        self.assertEqual(item["converged_at"], 1000)
        self.assertEqual(item["latency_seconds"], 0)
        self.assertTrue(item["met_deadline"])

    def test_older_does_not_reach(self):
        (item,) = evaluate_propagation_slo(
            [(1000, 100)],
            ["a", "b"],
            [[(1010, 90)], [(1010, 100)]],
            60,
        )
        self.assertIsNone(item["converged_at"])
        self.assertIsNone(item["latency_seconds"])
        self.assertFalse(item["met_deadline"])
        self.assertEqual(item["missing_domains"], ["a"])
        self.assertEqual(
            item["domain_reached_at"], [("a", None), ("b", 1010)]
        )

    def test_ambiguous_does_not_reach(self):
        (item,) = evaluate_propagation_slo(
            [(1000, 100)], ["a"], [[(1010, (100 + HALF) % MOD)]], 60
        )
        self.assertIsNone(item["converged_at"])
        self.assertEqual(item["missing_domains"], ["a"])

    def test_older_invalidates_domain_before_convergence(self):
        # a 域 1020 达到 -> 1030 回退（older）使域到达失效 -> 1050 重新达到；
        # b 域 1040 达到，故收敛时间为重达到时间 1050。
        (item,) = evaluate_propagation_slo(
            [(1000, 100)],
            ["a", "b"],
            [
                [(1020, 100), (1030, 90), (1050, 100)],
                [(1040, 100)],
            ],
            60,
        )
        self.assertEqual(item["converged_at"], 1050)
        self.assertEqual(item["latency_seconds"], 50)
        self.assertTrue(item["met_deadline"])
        self.assertEqual(
            item["domain_reached_at"], [("a", 1050), ("b", 1040)]
        )
        self.assertEqual(item["missing_domains"], [])

    def test_ambiguous_invalidates_domain(self):
        (item,) = evaluate_propagation_slo(
            [(1000, 100)],
            ["a", "b"],
            [
                [(1020, 100), (1030, (100 + HALF) % MOD), (1060, 100)],
                [(1040, 100)],
            ],
            60,
        )
        self.assertEqual(item["converged_at"], 1060)
        self.assertEqual(
            item["domain_reached_at"], [("a", 1060), ("b", 1040)]
        )

    def test_invalidation_without_rereach_missing_at_window_end(self):
        (item,) = evaluate_propagation_slo(
            [(1000, 100)],
            ["a", "b"],
            [
                [(1020, 100), (1030, 90)],
                [(1040, 100)],
            ],
            60,
        )
        self.assertIsNone(item["converged_at"])
        self.assertIsNone(item["latency_seconds"])
        self.assertFalse(item["met_deadline"])
        self.assertEqual(item["missing_domains"], ["a"])
        self.assertEqual(
            item["domain_reached_at"], [("a", None), ("b", 1040)]
        )

    def test_invalidation_after_convergence_freezes(self):
        # 1015 全部域已达到 -> 收敛冻结；a 域 1020 的回退不再改变结果
        (item,) = evaluate_propagation_slo(
            [(1000, 100)],
            ["a", "b"],
            [
                [(1010, 100), (1020, 90)],
                [(1015, 100)],
            ],
            60,
        )
        self.assertEqual(item["converged_at"], 1015)
        self.assertEqual(item["latency_seconds"], 15)
        self.assertTrue(item["met_deadline"])
        self.assertEqual(
            item["domain_reached_at"], [("a", 1010), ("b", 1015)]
        )
        self.assertEqual(item["missing_domains"], [])

    def test_non_arrived_older_does_not_invalidate(self):
        # 从未达到的副本出现 older 不影响（其域从未有效到达）；
        # 同域另一副本 1040 达到即可
        (item,) = evaluate_propagation_slo(
            [(1000, 100)],
            ["a", "a"],
            [
                [(1010, 90)],
                [(1040, 100)],
            ],
            60,
        )
        self.assertEqual(item["converged_at"], 1040)
        self.assertEqual(item["domain_reached_at"], [("a", 1040)])

    def test_other_replica_rereaches_invalidated_domain(self):
        # 域内 r1 到达后回退使域失效；由同域 r2 重新达到；
        # 另一域 b 直到 1060 才达到，故失效 / 重达到都发生在收敛之前
        (item,) = evaluate_propagation_slo(
            [(1000, 100)],
            ["a", "a", "b"],
            [
                [(1020, 100), (1030, 90)],
                [(1050, 100)],
                [(1060, 100)],
            ],
            60,
        )
        self.assertEqual(item["converged_at"], 1060)
        self.assertEqual(
            item["domain_reached_at"], [("a", 1050), ("b", 1060)]
        )

    def test_multi_replica_domain_earliest_arrival_wins(self):
        (item,) = evaluate_propagation_slo(
            [(1000, 100)],
            ["a", "a"],
            [
                [(1030, 100)],
                [(1010, 100)],
            ],
            60,
        )
        self.assertEqual(item["converged_at"], 1010)
        self.assertEqual(item["domain_reached_at"], [("a", 1010)])

    def test_atomic_same_timestamp_invalidation_and_rereach(self):
        # 同一时间戳：r1 回退使 a 域失效，同域 r2 同时达到 -> 重新建立，
        # b 域也在该时刻达到；结果不依赖事件排列（成批原子处理）。
        args = (
            [(1000, 100)],
            60,
        )
        events = [
            [(1020, 100), (1040, 90)],
            [(1040, 100)],
            [(1040, 100)],
        ]
        (item,) = evaluate_propagation_slo(
            args[0], ["a", "a", "b"], events, args[1]
        )
        self.assertEqual(item["converged_at"], 1040)
        self.assertEqual(
            item["domain_reached_at"], [("a", 1040), ("b", 1040)]
        )

        # 交换副本顺序（b 在前），结果一致
        (swapped,) = evaluate_propagation_slo(
            args[0],
            ["b", "a", "a"],
            [events[2], events[0], events[1]],
            args[1],
        )
        self.assertEqual(swapped["converged_at"], 1040)
        self.assertEqual(
            swapped["domain_reached_at"], [("a", 1040), ("b", 1040)]
        )

    def test_atomic_same_timestamp_invalidation_without_rereach(self):
        # 同刻 a 域失效、b 域首次达到：不能在该时刻收敛
        (item,) = evaluate_propagation_slo(
            [(1000, 100)],
            ["a", "b"],
            [
                [(1020, 100), (1050, 90)],
                [(1050, 100)],
            ],
            60,
        )
        self.assertIsNone(item["converged_at"])
        self.assertEqual(item["missing_domains"], ["a"])
        self.assertEqual(
            item["domain_reached_at"], [("a", None), ("b", 1050)]
        )

    def test_window_excludes_next_release_timestamp(self):
        # 2000 恰好是第二次发布时间：属于第二窗口，第一次发布窗口内无观测
        results = evaluate_propagation_slo(
            [(1000, 100), (2000, 101)],
            ["a"],
            [[(2000, 100), (2001, 101)]],
            60,
        )
        first, second = results
        self.assertIsNone(first["converged_at"])
        self.assertEqual(first["missing_domains"], ["a"])
        self.assertEqual(first["domain_reached_at"], [("a", None)])
        self.assertFalse(first["met_deadline"])
        self.assertEqual(second["converged_at"], 2001)
        self.assertEqual(second["latency_seconds"], 1)

    def test_observations_before_first_release_ignored(self):
        (item,) = evaluate_propagation_slo(
            [(1000, 100)], ["a"], [[(999, 100), (1001, 100)]], 60
        )
        self.assertEqual(item["converged_at"], 1001)

    def test_last_release_window_unbounded(self):
        (item,) = evaluate_propagation_slo(
            [(1000, 100)], ["a"], [[(9_999_999, 100)]], 60
        )
        self.assertEqual(item["converged_at"], 9_999_999)
        self.assertFalse(item["met_deadline"])
        self.assertEqual(item["latency_seconds"], 9_998_999)

    def test_no_observations(self):
        (item,) = evaluate_propagation_slo(
            [(1000, 100)], ["a"], [[]], 60
        )
        self.assertIsNone(item["converged_at"])
        self.assertIsNone(item["latency_seconds"])
        self.assertFalse(item["met_deadline"])
        self.assertEqual(item["domain_reached_at"], [("a", None)])
        self.assertEqual(item["missing_domains"], ["a"])

    def test_domains_sorted_in_output(self):
        (item,) = evaluate_propagation_slo(
            [(0, 1)],
            ["zeta", "alpha", "mid"],
            [[(10, 1)], [(2, 1)], [(6, 1)]],
            100,
        )
        self.assertEqual(item["converged_at"], 10)
        self.assertEqual(
            [d for d, _ in item["domain_reached_at"]],
            ["alpha", "mid", "zeta"],
        )
        self.assertEqual(
            item["domain_reached_at"],
            [("alpha", 2), ("mid", 6), ("zeta", 10)],
        )
        self.assertEqual(item["missing_domains"], [])

    def test_flap_multiple_times_rearrival_retimed(self):
        # 多次到达 / 回退：以最后一次重新达到的时间为准；
        # b 域直到 1060 才达到，全程没有提前收敛
        (item,) = evaluate_propagation_slo(
            [(1000, 100)],
            ["a", "b"],
            [
                [
                    (1010, 100),
                    (1020, 90),
                    (1030, 100),
                    (1040, 90),
                    (1060, 100),
                ],
                [(1060, 100)],
            ],
            60,
        )
        self.assertEqual(item["converged_at"], 1060)
        self.assertEqual(item["latency_seconds"], 60)
        self.assertEqual(
            item["domain_reached_at"], [("a", 1060), ("b", 1060)]
        )


# 路由层测试：直接以函数方式调用 FastAPI 端点（无需 httpx/TestClient）。
try:
    from fastapi import HTTPException as _HTTPException
    from pydantic import ValidationError as _ValidationError

    from app.main import app, propagation_slo
    from app.schemas import PropagationSloRequest

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover - CI 中 fastapi 必装
    _HAS_FASTAPI = False


@unittest.skipUnless(_HAS_FASTAPI, "fastapi 未安装")
class PropagationSloEndpointTests(unittest.TestCase):
    def _request(self, payload):
        return propagation_slo(PropagationSloRequest.model_validate(payload))

    def _base_payload(self):
        return {
            "deadline_seconds": 60,
            "releases": [
                {"released_at": 1000, "serial": 100},
                {"released_at": 2000, "serial": 101},
            ],
            "replicas": [
                {
                    "name": "ns-a",
                    "failure_domain": "dc-east",
                    "observations": [
                        {"timestamp": 1010, "serial": 100},
                        {"timestamp": 2005, "serial": 101},
                    ],
                },
                {
                    "name": "ns-b",
                    "failure_domain": "dc-west",
                    "observations": [
                        {"timestamp": 1040, "serial": 100},
                        {"timestamp": 2060, "serial": 101},
                    ],
                },
            ],
        }

    def test_endpoint_happy_path(self):
        resp = self._request(self._base_payload())
        self.assertEqual(resp.deadline_seconds, 60)
        self.assertEqual(len(resp.results), 2)
        first, second = resp.results

        self.assertEqual(first.released_at, 1000)
        self.assertEqual(first.serial, 100)
        self.assertEqual(first.converged_at, 1040)
        self.assertEqual(first.latency_seconds, 40)
        self.assertTrue(first.met_deadline)
        self.assertEqual(
            [(d.failure_domain, d.reached_at) for d in first.domain_reached_at],
            [("dc-east", 1010), ("dc-west", 1040)],
        )
        self.assertEqual(first.missing_domains, [])

        self.assertEqual(second.converged_at, 2060)
        self.assertEqual(second.latency_seconds, 60)
        self.assertTrue(second.met_deadline)

    def test_endpoint_missing_domains(self):
        payload = {
            "deadline_seconds": 60,
            "releases": [{"released_at": 1000, "serial": 100}],
            "replicas": [
                {
                    "name": "ns-a",
                    "failure_domain": "dc-east",
                    "observations": [
                        {"timestamp": 1010, "serial": 100},
                        {"timestamp": 1020, "serial": 90},
                    ],
                },
                {
                    "name": "ns-b",
                    "failure_domain": "dc-west",
                    "observations": [{"timestamp": 1015, "serial": 100}],
                },
            ],
        }
        (item,) = self._request(payload).results
        self.assertIsNone(item.converged_at)
        self.assertIsNone(item.latency_seconds)
        self.assertFalse(item.met_deadline)
        self.assertEqual(item.missing_domains, ["dc-east"])
        self.assertEqual(
            [(d.failure_domain, d.reached_at) for d in item.domain_reached_at],
            [("dc-east", None), ("dc-west", 1015)],
        )

    def test_endpoint_empty_observations_allowed(self):
        payload = {
            "deadline_seconds": 0,
            "releases": [{"released_at": 1000, "serial": 100}],
            "replicas": [
                {"name": "ns-a", "failure_domain": "dc", "observations": []},
            ],
        }
        (item,) = self._request(payload).results
        self.assertIsNone(item.converged_at)
        self.assertEqual(item.missing_domains, ["dc"])

    def test_endpoint_duplicate_replica_name_422(self):
        payload = self._base_payload()
        payload["replicas"][1]["name"] = "ns-a"
        with self.assertRaises(_HTTPException) as ctx:
            self._request(payload)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(ctx.exception.detail["error"], "duplicate_replica_name")

    def test_endpoint_duplicate_release_timestamp_422(self):
        payload = self._base_payload()
        payload["releases"][1]["released_at"] = 1000
        with self.assertRaises(_HTTPException) as ctx:
            self._request(payload)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(
            ctx.exception.detail["error"], "duplicate_release_timestamp"
        )

    def test_endpoint_release_out_of_order_422(self):
        payload = self._base_payload()
        payload["releases"][1]["released_at"] = 999
        with self.assertRaises(_HTTPException) as ctx:
            self._request(payload)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(
            ctx.exception.detail["error"], "release_timestamp_out_of_order"
        )

    def test_endpoint_duplicate_observation_timestamp_422(self):
        payload = self._base_payload()
        payload["replicas"][0]["observations"][1]["timestamp"] = 1010
        with self.assertRaises(_HTTPException) as ctx:
            self._request(payload)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(
            ctx.exception.detail["error"], "duplicate_observation_timestamp"
        )
        self.assertEqual(ctx.exception.detail["replica"], "ns-a")

    def test_endpoint_observation_out_of_order_422(self):
        payload = self._base_payload()
        payload["replicas"][0]["observations"][1]["timestamp"] = 1005
        with self.assertRaises(_HTTPException) as ctx:
            self._request(payload)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(
            ctx.exception.detail["error"], "observation_timestamp_out_of_order"
        )

    def test_request_validation_422(self):
        base = self._base_payload()

        def mutated(**changes):
            import copy

            payload = copy.deepcopy(base)
            for key, value in changes.items():
                payload[key] = value
            return payload

        invalid_payloads = [
            mutated(releases=[]),
            mutated(replicas=[]),
            mutated(deadline_seconds=-1),
            mutated(deadline_seconds=True),
            mutated(deadline_seconds=1.5),
            mutated(deadline_seconds="60"),
            # 发布时间戳 / 序列号非法
            mutated(releases=[{"released_at": -1, "serial": 1}]),
            mutated(releases=[{"released_at": True, "serial": 1}]),
            mutated(releases=[{"released_at": 1.5, "serial": 1}]),
            mutated(releases=[{"released_at": 0, "serial": -1}]),
            mutated(releases=[{"released_at": 0, "serial": 2**32}]),
            mutated(releases=[{"released_at": 0, "serial": "1"}]),
            # 副本名 / 故障域
            mutated(
                replicas=[
                    {
                        "name": "  ",
                        "failure_domain": "dc",
                        "observations": [{"timestamp": 1, "serial": 1}],
                    }
                ]
            ),
            mutated(
                replicas=[
                    {
                        "name": "ns-a",
                        "failure_domain": "   ",
                        "observations": [{"timestamp": 1, "serial": 1}],
                    }
                ]
            ),
            # 观测时间戳 / 序列号非法
            mutated(
                replicas=[
                    {
                        "name": "ns-a",
                        "failure_domain": "dc",
                        "observations": [{"timestamp": -1, "serial": 1}],
                    }
                ]
            ),
            mutated(
                replicas=[
                    {
                        "name": "ns-a",
                        "failure_domain": "dc",
                        "observations": [{"timestamp": 1.5, "serial": 1}],
                    }
                ]
            ),
            mutated(
                replicas=[
                    {
                        "name": "ns-a",
                        "failure_domain": "dc",
                        "observations": [{"timestamp": 1, "serial": 2**32}],
                    }
                ]
            ),
        ]
        for payload in invalid_payloads:
            with self.assertRaises(
                _ValidationError, msg=f"payload={payload}"
            ):
                PropagationSloRequest.model_validate(payload)

    def test_openapi_lists_propagation_slo_with_examples(self):
        schema = app.openapi()
        self.assertIn("/propagation-slo", schema["paths"])
        self.assertIn("post", schema["paths"]["/propagation-slo"])
        request_schema = schema["components"]["schemas"]["PropagationSloRequest"]
        self.assertIn("examples", request_schema)
        self.assertGreaterEqual(len(request_schema["examples"]), 1)
        for example in request_schema["examples"]:
            PropagationSloRequest.model_validate(example)


if __name__ == "__main__":
    unittest.main()
