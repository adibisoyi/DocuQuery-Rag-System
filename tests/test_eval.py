from eval.run_eval import load_eval_dataset


def test_eval_dataset_loads() -> None:
    dataset = load_eval_dataset("eval/eval_dataset.json")
    assert len(dataset) >= 1
    assert "question" in dataset[0]
    assert "expected_source" in dataset[0]
    assert "expected_answer_contains" in dataset[0]