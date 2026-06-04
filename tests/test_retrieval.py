import numpy as np
from unittest.mock import patch, MagicMock

from src.retrieval import compute_bge_m3_embeddings


def _fake_response(vectors):
    """Build a mock requests.Response returning a JSON list of vectors."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = vectors
    resp.raise_for_status = MagicMock()
    return resp


def test_returns_correct_shape():
    texts = ["hello world", "goodbye world"]
    fake_vectors = [[0.1] * 1024, [0.2] * 1024]
    with patch("src.retrieval.requests.post", return_value=_fake_response(fake_vectors)):
        result = compute_bge_m3_embeddings(texts, checkpoint_path=None)
    assert result.shape == (2, 1024)
    assert result.dtype == np.float32


def test_uses_bge_m3_endpoint():
    texts = ["a"]
    fake_vectors = [[0.0] * 1024]
    with patch("src.retrieval.requests.post", return_value=_fake_response(fake_vectors)) as mock_post:
        compute_bge_m3_embeddings(texts, checkpoint_path=None)
    called_url = mock_post.call_args.args[0]
    assert called_url == "http://192.168.31.246:5000/embed"
    payload = mock_post.call_args.kwargs["json"]
    assert payload == {"inputs": ["a"]}


def test_returns_zero_vectors_on_final_failure():
    texts = ["x"]
    with patch("src.retrieval.requests.post", side_effect=ConnectionError("server down")):
        result = compute_bge_m3_embeddings(texts, checkpoint_path=None)
    assert result.shape == (1, 1024)
    assert np.all(result == 0.0)


def test_preserves_input_order():
    texts = ["a", "b", "c"]
    fake_vectors = [[1.0] + [0.0] * 1023, [2.0] + [0.0] * 1023, [3.0] + [0.0] * 1023]
    with patch("src.retrieval.requests.post", return_value=_fake_response(fake_vectors)):
        result = compute_bge_m3_embeddings(texts, checkpoint_path=None)
    assert result[0, 0] == 1.0
    assert result[1, 0] == 2.0
    assert result[2, 0] == 3.0


def test_4xx_response_does_not_retry():
    texts = ["a", "b"]
    bad_resp = MagicMock()
    bad_resp.status_code = 422
    bad_resp.text = "malformed payload"
    bad_resp.raise_for_status = MagicMock()
    with patch("src.retrieval.requests.post", return_value=bad_resp) as mock_post:
        result = compute_bge_m3_embeddings(texts, checkpoint_path=None)
    assert mock_post.call_count == 1, "4xx should not be retried"
    assert result.shape == (2, 1024)
    assert np.all(result == 0.0)
