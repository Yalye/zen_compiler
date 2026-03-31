//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "StandaloneDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Standalone_FooOp : Standalone_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `standalone.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = standalone.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def FilterPointOp : Standalone_Op<"filter_point", [Pure]> {
    let summary = "Filters a 3D point based on a threshold.";
    let description = [{
        此算子接收一个 3D 向量（点）和一个阈值，如果点强度超过阈值则保留。
    }];

    // 输入：一个 3D 向量（Tensor）和一个浮点数阈值
    let arguments = (ins F32Tensor:$input_point, F32:$threshold);

    // 输出：过滤后的 3D 向量
    let results = (outs F32Tensor:$output_point);

    // 允许自动生成 C++ 声明
    // let hasCustomAssemblyFormat = 1;

    let assemblyFormat = " `(` $input_point `,` $threshold `)` attr-dict `:` `(` type($input_point) `,` type($threshold) `)` `->` type($output_point) ";
}

def HelloWorldOp : Standalone_Op<"hello_world"> {
    let summary = "my first mlir op";
    let description = [{
        这个算子不接收参数，也不返回结果，仅用于证明 Dialect 跑通了。
    }];

    let arguments = (ins);
    let results = (outs);
}


#endif // STANDALONE_OPS
