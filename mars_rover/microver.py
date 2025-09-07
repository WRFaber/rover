# Micropython compatible methods for inference with assumptions that policynet_quantized.py is available

def relu(x):
    return [max(0, xi) for xi in x]

def softmax(x):
    import math
    exp_x = [math.exp(xi / 256.0) for xi in x]
    sum_exp = sum(exp_x)
    return [xi / sum_exp for xi in exp_x]

def matmul(x, weights, bias, w_scale, b_scale):
    out = []
    for i in range(len(weights)):
        acc = 0
        for j in range(len(x)):
            acc += x[j] * weights[i][j]
        acc = acc * w_scale + bias[i] * b_scale
        out.append(acc)
    return out

def policynet_forward(x):
    from policynet_quantized import (
        fc1_weight, fc1_bias, fc1_weight_scale, fc1_bias_scale,
        fc2_weight, fc2_bias, fc2_weight_scale, fc2_bias_scale,
        fc3_weight, fc3_bias, fc3_weight_scale, fc3_bias_scale
    )

    x = matmul(x, fc1_weight, fc1_bias, fc1_weight_scale, fc1_bias_scale)
    x = relu(x)
    x = matmul(x, fc2_weight, fc2_bias, fc2_weight_scale, fc2_bias_scale)
    x = relu(x)
    x = matmul(x, fc3_weight, fc3_bias, fc3_weight_scale, fc3_bias_scale)
    x = softmax(x)
    return x