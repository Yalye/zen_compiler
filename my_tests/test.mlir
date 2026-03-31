// test.mlir
module {
  func.func @main() {
    // 调用你的新算子
    "standalone.hello_world"() : () -> ()
    return
  }
}