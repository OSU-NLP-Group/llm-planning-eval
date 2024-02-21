from func_timeout import func_timeout

def exec_py(py_str):
    _locals = locals()
    func_timeout(10.0, exec, args=(py_str, None, _locals))

    if "answer" in _locals:
        answer = _locals["answer"]
    else:
        answer = None

    return answer