#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;

// 注意：如果你的 Dialect 命名空间在 TableGen 里定义为 standalone，
// 那么 FilterPointOp 应该在 mlir::standalone 命名空间下。
using namespace mlir::standalone; 

// 1. 定义降级模式
struct FilterPointLowering : public OpRewritePattern<FilterPointOp> {
  using OpRewritePattern<FilterPointOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FilterPointOp op,
                              PatternRewriter &rewriter) const override {
  auto loc = op.getLoc();
  Value input = op.getInputPoint();
  Value threshold = op.getThreshold();

  // 1. 获取输入类型并转换 (新版写法：llvm::cast)
  auto inputType = llvm::cast<RankedTensorType>(input.getType());

  // 2. 将标量广播为张量 (使用 tensor.splat，这是最稳妥的标量转张量方法)
  Value broadcastThreshold = rewriter.create<tensor::SplatOp>(
      loc, inputType, threshold);

  // 3. 比较 (这时两边都是 tensor<3xf32> 了)
  // 结果类型会自动推导为 tensor<3xi1>
  Value isGreater = rewriter.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OGT, input, broadcastThreshold);
  
  // 4. 创建广播后的 0 值张量
  Value zeroScalar = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF32FloatAttr(0.0f));
  Value broadcastZero = rewriter.create<tensor::SplatOp>(
      loc, inputType, zeroScalar);

  // 5. 选择
  rewriter.replaceOpWithNewOp<arith::SelectOp>(op, isGreater, input, broadcastZero);

  return success();
}
};



// 2. 定义 Pass
// 修正：PassWrapper 的第二个参数通常是具体的 Base 类，
// 既然你没有用 TableGen 生成 Pass 基类，我们直接用 OperationPass<ModuleOp>
struct TestLoweringPass : public PassWrapper<TestLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLoweringPass)

  // 关键修改：声明本 Pass 依赖 Tensor 方言
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, arith::ArithDialect>();
  }

  StringRef getArgument() const final { return "test-zen-lowering"; }
  StringRef getDescription() const final { return "降级机器人过滤算子"; }

  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FilterPointLowering>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

// 3. 注册函数
void registerTestPass() {
    PassRegistration<TestLoweringPass>();
}