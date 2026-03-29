module {
  func.func @test_filter(%input: tensor<3xf32>) -> tensor<3xf32> {
    %cst = arith.constant 0.8 : f32
    // 调用你刚定义的算子
    %result = standalone.filter_point(%input, %cst) : (tensor<3xf32>, f32) -> tensor<3xf32>
    return %result : tensor<3xf32>
  }
}