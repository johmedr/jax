#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// It requires these headers, but does not include them.
// NOLINTNEXTLINE(misc-include-cleaner)
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// NOLINTNEXTLINE(misc-include-cleaner)
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "absl/log/check.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/include/mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/include/mlir/IR/AffineExpr.h"
#include "mlir/include/mlir/IR/Attributes.h"
#include "mlir/include/mlir/IR/Block.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Region.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_CANONICALIZEMOSAICPASS
#define GEN_PASS_DEF_CANONICALIZEMOSAICPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

LogicalResult tpu_matmul_rule(tpu::MatmulOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op.getOperation());

  auto lhs = op.getLhs();
  auto rhs = op.getRhs();
  auto acc = op.getAcc();

  const VectorType lhs_ty = lhs.getType();
  const VectorType rhs_ty = rhs.getType();
  const VectorType acc_ty = acc.getType();

  auto lhs_element_type = lhs_ty.getElementType();
  auto rhs_element_type = rhs_ty.getElementType();
  auto acc_element_type = acc_ty.getElementType();

  auto extsi_sitofp = [&builder, &op](TypedValue<VectorType> element) {
    const VectorType ty = element.getType();
    auto shape = ty.getShape();
    CHECK(ty.getElementType().isInteger());
    TypedValue<VectorType> ext_ele;
    if (ty.getElementType().getIntOrFloatBitWidth() == 32) {
      ext_ele = element;
    } else {
      ext_ele = cast<TypedValue<VectorType>>(
          builder
              .create<arith::ExtSIOp>(
                  VectorType::get(shape, builder.getI32Type()), element)
              .getResult());
    }
    // TODO(mvoz): Go to bf16 when hardware supported, requires adding support
    // for 16 bitwidth in extsiop in infer/apply.
    auto ele_as_fp = builder.create<arith::SIToFPOp>(
        op.getLoc(), VectorType::get(shape, builder.getF32Type()), ext_ele);
    return ele_as_fp;
  };

  if (lhs_element_type != rhs_element_type) {
    if (lhs_element_type.isInteger() && rhs_element_type.isInteger()) {
      // TODO(mvoz): Add support for mixed int/int matmul.
      op->emitOpError("Mix int/int - NYI");
      return failure();
    }
    if (acc_element_type.isInteger()) {
      // TODO(mvoz): Add support for mixed int/float matmul with int acc.
      // Should be pretty straightforward.
      op->emitOpError("acc is int in mixed matmul. Expected float.");
      return failure();
    }
    if (lhs_element_type.isInteger()) {
      auto float_lhs = extsi_sitofp(lhs);
      op->setOperand(0, float_lhs);
    }
    if (rhs_element_type.isInteger()) {
      auto float_rhs = extsi_sitofp(rhs);
      op->setOperand(1, float_rhs);
    }
  }
  // TODO(mvoz): Add more invariants.
  if (acc_element_type.isInteger()) {
    if (!op.getLhs().getType().getElementType().isInteger()) {
      op->emitOpError("int acc with float lhs. Expected int lhs.");
      return failure();
    }
    if (!op.getRhs().getType().getElementType().isInteger()) {
      op->emitOpError("int acc with float rhs. Expected int rhs.");
      return failure();
    }
  } else {
    if (op.getLhs().getType().getElementType().isInteger()) {
      op->emitOpError("float acc with int lhs. Expected float lhs.");
      return failure();
    }
    if (op.getRhs().getType().getElementType().isInteger()) {
      op->emitOpError("float acc with int rhs. Expected float rhs.");
      return failure();
    }
  }
  return success();
};

LogicalResult canonicalize_elementwise(int hardware_generation_,
                                       Operation &op) {
  OpBuilder builder(&op);
  auto operands = op.getOperands();
  auto res_ty = dyn_cast<VectorType>(op.getResult(0).getType());
  if (op.getNumResults() != 1) {
    op.emitOpError("Invariant violated: Unexpected number of results");
    return failure();
  }
  if (!res_ty) {
    // scalar
    // TODO(mvoz): Add canonicalization and invariants for scalar elementwise
    // ops.
    return success();
  }
  auto shape = res_ty.getShape();
  std::vector<Value> new_operands;
  new_operands.reserve(operands.size());

  bool should_rewrite_op = false;
  auto target_f32_ty = VectorType::get(shape, builder.getF32Type());
  for (int i = 0; i < operands.size(); ++i) {
    auto operand = operands[i];
    auto ty = dyn_cast<VectorType>(operand.getType());
    if (ty) {
      if (ty.getShape() != shape) {
        // Should already be checked my MLIR verification, but let's be safe.
        op.emitOpError("Mismatched shapes in elementwise op.");
        return failure();
      }
      auto element_type = ty.getElementType();
      // PowFOp and DivFOp do not seem to be supported in bf16 on later
      // hardware.
      bool needs_cast = hardware_generation_ <= 5 || isa<math::PowFOp>(op) ||
                        isa<arith::DivFOp>(op);
      if (needs_cast && element_type.isBF16()) {
        auto target_f32 =
            builder.create<arith::ExtFOp>(op.getLoc(), target_f32_ty, operand)
                .getResult();
        should_rewrite_op = true;
        new_operands.push_back(target_f32);
      } else {
        new_operands.push_back(operand);
      }
    } else {
      // Should already be checked my MLIR verification, but let's be safe.
      op.emitOpError("MLIR unsupported - mix scalar and vec elementwise ops");
      return failure();
    }
  }
  if (should_rewrite_op) {
    auto result_ty = dyn_cast<VectorType>(op.getResult(0).getType());
    if (!result_ty) {
      op.emitOpError("Not implemented: Unexpected result type");
      return failure();
    }
    auto result_element_type = result_ty.getElementType();
    if (!result_element_type.isF32() && !result_element_type.isBF16()) {
      op.emitOpError("Not implemented: Unexpected result element type");
      return failure();
    }
    // Do the new op in f32, then truncate to the original element type.
    auto new_op = builder.create(op.getLoc(), op.getName().getIdentifier(),
                                 new_operands, target_f32_ty);
    new_op = builder.create<arith::TruncFOp>(op.getLoc(), res_ty,
                                             new_op->getResult(0));
    op.replaceAllUsesWith(new_op);
    op.erase();
  }
  return success();
}

LogicalResult canonicalize_multi_dim_reduction(int hardware_generation,
                                               Operation &operation) {
  ImplicitLocOpBuilder builder(operation.getLoc(), &operation);
  auto op = cast<vector::MultiDimReductionOp>(operation);
  auto source_ty = op.getSourceVectorType();
  auto result_ty = dyn_cast<VectorType>(op.getDestType());
  if (!result_ty) {
    return op->emitOpError() << "Only vector reductions supported";
  }

  auto element_type = source_ty.getElementType();
  if (element_type.isF32()) {
    return success();
  } else if (element_type.isBF16()) {
    bool reduces_sublanes = false;
    for (int64_t dim : op.getReductionDims()) {
      if (dim == source_ty.getRank() - 2) {
        reduces_sublanes = true;
      }
    }
    if (hardware_generation <= 5 || reduces_sublanes) {
      auto new_source = builder.create<arith::ExtFOp>(
          VectorType::get(source_ty.getShape(), builder.getF32Type()),
          op.getSource());

      auto result_ty_f32 =
          VectorType::get(result_ty.getShape(), builder.getF32Type());
      auto acc_ext = builder.create<arith::ExtFOp>(result_ty_f32, op.getAcc());
      Value new_acc = acc_ext.getResult();
      // Try to constant fold.
      if (auto const_acc = op.getAcc().getDefiningOp<arith::ConstantOp>()) {
        auto result =
            acc_ext.fold(arith::ExtFOp::FoldAdaptor(const_acc.getValue()));
        if (!result.isNull() && result.is<Attribute>()) {
          acc_ext->erase();
          new_acc = builder.create<arith::ConstantOp>(
              op.getLoc(), result_ty_f32,
              cast<TypedAttr>(result.get<Attribute>()));
        }
      }
      auto new_op = builder.create<vector::MultiDimReductionOp>(
          op.getLoc(), new_acc.getType(), op.getKindAttr(), new_source, new_acc,
          DenseI64ArrayAttr::get(builder.getContext(), op.getReductionDims()));
      auto new_result = builder.create<arith::TruncFOp>(op.getLoc(), result_ty,
                                                        new_op.getResult());
      op.replaceAllUsesWith(new_result.getResult());
      op.erase();
    }
    return success();
  }
  return failure();
}

LogicalResult canonicalize_matmul(int hardware_generation, Operation &op) {
  auto matmul_op = dyn_cast<tpu::MatmulOp>(op);
  if (!matmul_op) {
    op.emitOpError("Invariant violated: Not a matmul");
    return failure();
  }
  return tpu_matmul_rule(matmul_op);
};

LogicalResult canonicalize_contraction(int hardware_generation, Operation &op) {
  auto contraction_op = dyn_cast<vector::ContractionOp>(op);
  if (!contraction_op) {
    op.emitOpError("Invariant violated: Not a contraction");
    return failure();
  }
  // Rewrite the contraction as a matmul
  auto lhs = contraction_op.getLhs();
  auto rhs = contraction_op.getRhs();
  auto acc = contraction_op.getAcc();
  VectorType acc_ty;
  if (!(acc_ty = dyn_cast<VectorType>(acc.getType()))) {
    contraction_op->emitOpError("Not implemented: acc must be a vector");
    return failure();
  }

  if (contraction_op.getKind() != vector::CombiningKind::ADD) {
    contraction_op->emitOpError("Only ADD supported");
    return failure();
  }

  ImplicitLocOpBuilder builder(contraction_op->getLoc(),
                               contraction_op.getOperation());

  MLIRContext *const mlir_ctx = contraction_op->getContext();

  auto getMapAttr = [&](const unsigned first, const unsigned second) {
    return AffineMapAttr::get(AffineMap::get(
        3, 0,
        {getAffineDimExpr(first, mlir_ctx), getAffineDimExpr(second, mlir_ctx)},
        mlir_ctx));
  };

  const ArrayAttr matmul_indexing_maps = builder.getArrayAttr(
      {getMapAttr(0, 2), getMapAttr(2, 1), getMapAttr(0, 1)});
  const ArrayAttr matmul_indexing_maps_transposed = builder.getArrayAttr(
      {getMapAttr(0, 2), getMapAttr(1, 2), getMapAttr(0, 1)});
  const auto indexing_maps = contraction_op.getIndexingMaps();
  if (indexing_maps != matmul_indexing_maps &&
      indexing_maps != matmul_indexing_maps_transposed) {
    return contraction_op->emitOpError(
        "Not implemented: Non-matmul or unsupported indexing_maps");
  }
  const bool transpose_rhs = indexing_maps == matmul_indexing_maps_transposed;

  const ArrayAttr matmul_iterator_types =
      builder.getArrayAttr({builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::parallel),
                            builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::parallel),
                            builder.getAttr<vector::IteratorTypeAttr>(
                                vector::IteratorType::reduction)});
  if (contraction_op->getAttr("iterator_types") != matmul_iterator_types) {
    return contraction_op->emitOpError(
        "Not implemented: Non-matmul iterator_types");
  }
  const tpu::ContractPrecisionAttr precision_attr =  // May be null
      contraction_op->getAttrOfType<tpu::ContractPrecisionAttr>("precision");
  auto matmul_op = builder.create<tpu::MatmulOp>(
      contraction_op->getLoc(), acc_ty, lhs, rhs, acc,
      /*transpose_lhs=*/false, transpose_rhs, precision_attr);
  contraction_op.replaceAllUsesWith(matmul_op.getResult());
  contraction_op.erase();
  auto result = tpu_matmul_rule(matmul_op);
  return result;
}

LogicalResult canonicalize_extract(int hardware_generation, Operation &raw_op) {
  auto op = dyn_cast<vector::ExtractOp>(raw_op);
  Type result_ty = op.getResult().getType();
  if (!isa<VectorType>(result_ty)) {
    bool is_supported = result_ty.isSignlessIntOrFloat() &&
                        result_ty.getIntOrFloatBitWidth() == 32;
    if (!is_supported) {
      return op.emitOpError(
          "Only 32-bit scalar vector.extracts supported. Cast your input to a "
          "32-bit type first.");
    }
  }
  return success();
}

LogicalResult canonicalize_select(int hardware_generation, Operation &raw_op) {
  auto op = dyn_cast<arith::SelectOp>(raw_op);
  if (!isa<VectorType>(op.getType()) ||
      isa<VectorType>(op.getCondition().getType())) {
    return success();
  }
  // Canonicalize `i1 ? v1 : v2` -> `broadcast(i1) ? v1 : v2`.
  ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
  auto cond_ty = VectorType::get(cast<VectorType>(op.getType()).getShape(),
                                 op.getCondition().getType());
  auto cond = builder.create<vector::BroadcastOp>(cond_ty, op.getCondition());
  auto new_op = builder.create<arith::SelectOp>(
      op.getLoc(), cond, op.getTrueValue(), op.getFalseValue());
  op.replaceAllUsesWith(new_op.getResult());
  op.erase();
  return success();
}

LogicalResult canonicalize_repeat(int hardware_generation, Operation &raw_op) {
  auto op = dyn_cast<tpu::RepeatOp>(raw_op);
  if (!isa<VectorType>(op.getType())) {
    return success();
  }
  auto operand = op.getSource();
  std::vector<Value> operands;
  operands.reserve(op.getTimes());
  for (int64_t i = 0; i < op.getTimes(); i++) {
    operands.push_back(operand);
  }
  ImplicitLocOpBuilder builder(op->getLoc(), op.getOperation());
  auto concat = builder.create<tpu::ConcatenateOp>(op.getLoc(), op.getType(),
                                                   operands, op.getDimension());
  op.replaceAllUsesWith(concat.getResult());
  op.erase();
  return success();
}

using canonicalize_rule_type =
    std::function<LogicalResult(int hardware_generation, Operation &op)>;

const llvm::StringMap<canonicalize_rule_type> &rules() {
  static auto rules = new llvm::StringMap<canonicalize_rule_type>{
      {tpu::MatmulOp::getOperationName(), canonicalize_matmul},
      {vector::ContractionOp::getOperationName(), canonicalize_contraction},
      {vector::ContractionOp::getOperationName(), canonicalize_extract},
      {vector::MultiDimReductionOp::getOperationName(),
       canonicalize_multi_dim_reduction},
      {arith::SelectOp::getOperationName(), canonicalize_select},
      {tpu::RepeatOp::getOperationName(), canonicalize_repeat}};
  return *rules;
}

const llvm::StringSet<> &elementwise_convertible_ops() {
  static auto ops = new llvm::StringSet<>{arith::MulFOp::getOperationName(),
                                          arith::DivFOp::getOperationName(),
                                          arith::AddFOp::getOperationName(),
                                          arith::SubFOp::getOperationName(),
                                          arith::MaximumFOp::getOperationName(),
                                          arith::MinimumFOp::getOperationName(),
                                          math::PowFOp::getOperationName()};
  return *ops;
}

class MosaicCanonicalizer {
 public:
  MosaicCanonicalizer(int hardware_generation)
      : hardware_generation_(hardware_generation) {}

  int hardware_generation_;

  LogicalResult canonicalize(func::FuncOp op) {
    if (!op.getBody().hasOneBlock()) {
      op.emitOpError("Only one block functions supported");
      return failure();
    }
    return canonicalizeBlock(op.getBody().front());
  }

  LogicalResult canonicalizeBlock(Block &block) {
    // make_early_inc_range is utilized due to op mutation.
    for (Operation &any_op : make_early_inc_range(block)) {
      if (canonicalizeOp(any_op).failed()) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult canonicalizeOp(Operation &any_op) {
    // We must iterate over the op first, because canonicalization can cause
    // us to .erase() an op, and accessing getRegions on it after is not sound.
    // Invariant - top level ops with regions may never be invalidated.
    for (Region &region : any_op.getRegions()) {
      for (Block &block : region) {
        if (canonicalizeBlock(block).failed()) {
          return failure();
        }
      }
    }
    if (elementwise_convertible_ops().contains(
            any_op.getName().getStringRef())) {
      return canonicalize_elementwise(hardware_generation_, any_op);
    }
    if (auto rule_it = rules().find(any_op.getName().getStringRef());
        rule_it != rules().end()) {
      const canonicalize_rule_type &rule = rule_it->getValue();
      return rule(hardware_generation_, any_op);
    }
    return success();
  }
};

struct CanonicalizeMosaicPass
    : public impl::CanonicalizeMosaicPassBase<CanonicalizeMosaicPass> {
  CanonicalizeMosaicPass(int hardware_generation) {
    this->hardware_generation = hardware_generation;
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MosaicCanonicalizer vlc(hardware_generation);
    if (vlc.canonicalize(func).failed()) {
      signalPassFailure();
    }
  };
};

std::unique_ptr<OperationPass<func::FuncOp>> createCanonicalizeMosaicPass(
    int hardware_generation) {
  return std::make_unique<CanonicalizeMosaicPass>(hardware_generation);
}

}  // namespace mlir::tpu
