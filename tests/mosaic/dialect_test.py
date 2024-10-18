# Copyright 2024 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""(Deviceless) tests for the Mosaic GPU MLIR dialect."""

from absl.testing import parameterized
from jax._src import config
from jax._src import test_util as jtu
from jax.experimental.mosaic.gpu import mosaic_gpu_dialect as mgpu
from mlir import ir


config.parse_flags_with_absl()


def _make_ir_context():
  context = ir.Context()
  mgpu.register_dialect(context)
  return context


class DialectTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()

  def test_initialize_barrier_op_enforces_relevant_invariants(self):
    with self.subTest("memref_must_wrap_barriers"):
      with ir.InsertionPoint(self.module.body):
        op = mgpu.initialize_barrier(
            ir.MemRefType.get((1, 2), ir.F32Type.get()),
            arrival_count=1,
        )

      with self.assertRaisesRegex(
          ir.MLIRError, "must be memref of barrier values"
      ):
        self.module.operation.verify()
      op.owner.operation.erase()

    with self.subTest("arrival_count_must_be_strictly_positive"):
      with ir.InsertionPoint(self.module.body):
        op = mgpu.initialize_barrier(
            ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
            arrival_count=0,
        )
      with self.assertRaisesRegex(ir.MLIRError, "value is positive"):
        self.module.operation.verify()
      op.owner.operation.erase()

    with self.subTest("wrapping_barriers_with_a_positive_arrival_count_passes"):
      with ir.InsertionPoint(self.module.body):
        mgpu.initialize_barrier(
            ir.MemRefType.get((1, 2), ir.Type.parse("!mosaic_gpu.barrier")),
            arrival_count=1,
        )

      self.assertTrue(self.module.operation.verify())


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())
