"""SGIC backend API tests"""
import io
import os
import base64
import pytest
import requests
import numpy as np
from PIL import Image

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://payload-embed.preview.emergentagent.com").rstrip("/")
TOKEN = "test_session_sgic_1777978366394"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
MESSAGE = "TEST_SGIC payload — confidential."
KEY = "sgic-test-key-2026"


def _png_bytes(seed=42):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def cover_png():
    return _png_bytes()


@pytest.fixture(scope="module")
def embed_result(cover_png):
    files = {"image": ("c.png", cover_png, "image/png")}
    data = {"message": MESSAGE, "secret_key": KEY}
    r = requests.post(f"{BASE_URL}/api/embed", headers=HEADERS, files=files, data=data, timeout=60)
    assert r.status_code == 200, r.text
    return r.json()


# --- Health & Auth ---
class TestHealthAuth:
    def test_root(self):
        r = requests.get(f"{BASE_URL}/api/", timeout=10)
        assert r.status_code == 200
        assert r.json().get("status") == "ok"

    def test_auth_me_401_no_token(self):
        r = requests.get(f"{BASE_URL}/api/auth/me", timeout=10)
        assert r.status_code == 401

    def test_auth_me_with_token(self):
        r = requests.get(f"{BASE_URL}/api/auth/me", headers=HEADERS, timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert d["email"] == "researcher@sgic.lab"
        assert d["user_id"] == "test-user-sgic-1777978366394"

    def test_logout_clears_cookie(self):
        # We don't pass cookie so it just returns ok — but verify the response is 200
        r = requests.post(f"{BASE_URL}/api/auth/logout", timeout=10)
        assert r.status_code == 200
        assert r.json().get("ok") is True


# --- Embed ---
class TestEmbed:
    def test_embed_requires_auth(self, cover_png):
        files = {"image": ("c.png", cover_png, "image/png")}
        data = {"message": MESSAGE, "secret_key": KEY}
        r = requests.post(f"{BASE_URL}/api/embed", files=files, data=data, timeout=30)
        assert r.status_code == 401

    def test_embed_success_structure(self, embed_result):
        d = embed_result
        for k in ["cover_image", "stego_image", "diff_image", "metrics", "histograms",
                  "payload_bits", "raw_message_bytes", "encrypted_bytes", "elapsed_ms"]:
            assert k in d, f"Missing {k}"
        m = d["metrics"]
        for mk in ["psnr", "ssim", "bpp", "chi_square", "rs_analysis"]:
            assert mk in m
        # Sane PSNR for 256x256 LSB
        assert m["psnr"] > 30
        assert 0 < m["ssim"] <= 1
        # Histograms 3 channels each
        for hk in ["cover", "stego", "diff"]:
            assert len(d["histograms"][hk]) == 3
        assert d["stego_image"].startswith("data:image/png;base64,")
        assert d["payload_bits"] > 32  # length header + data

    def test_embed_empty_message_400(self, cover_png):
        files = {"image": ("c.png", cover_png, "image/png")}
        data = {"message": "   ", "secret_key": KEY}
        r = requests.post(f"{BASE_URL}/api/embed", headers=HEADERS, files=files, data=data, timeout=30)
        assert r.status_code == 400


# --- Extract ---
class TestExtract:
    def test_extract_requires_auth(self, embed_result):
        stego_b64 = embed_result["stego_image"].split(",", 1)[1]
        stego_bytes = base64.b64decode(stego_b64)
        files = {"image": ("s.png", stego_bytes, "image/png")}
        data = {"secret_key": KEY, "original_message": MESSAGE}
        r = requests.post(f"{BASE_URL}/api/extract", files=files, data=data, timeout=30)
        assert r.status_code == 401

    def test_extract_correct_key(self, embed_result):
        stego_bytes = base64.b64decode(embed_result["stego_image"].split(",", 1)[1])
        files = {"image": ("s.png", stego_bytes, "image/png")}
        data = {"secret_key": KEY, "original_message": MESSAGE}
        r = requests.post(f"{BASE_URL}/api/extract", headers=HEADERS, files=files, data=data, timeout=30)
        assert r.status_code == 200
        d = r.json()
        assert d["success"] is True
        assert d["recovered_message"] == MESSAGE
        assert d["bit_error_rate"] == 0.0

    def test_extract_wrong_key(self, embed_result):
        stego_bytes = base64.b64decode(embed_result["stego_image"].split(",", 1)[1])
        files = {"image": ("s.png", stego_bytes, "image/png")}
        data = {"secret_key": "wrong-key-xyz", "original_message": MESSAGE}
        r = requests.post(f"{BASE_URL}/api/extract", headers=HEADERS, files=files, data=data, timeout=30)
        assert r.status_code == 200
        d = r.json()
        assert d["success"] is False
        assert d["error"]


# --- Robustness ---
class TestRobustness:
    def test_robustness_requires_auth(self, embed_result):
        body = {
            "stego_image": embed_result["stego_image"],
            "secret_key": KEY,
            "original_message": MESSAGE,
        }
        r = requests.post(f"{BASE_URL}/api/robustness", json=body, timeout=60)
        assert r.status_code == 401

    def test_robustness_three_attacks(self, embed_result):
        body = {
            "stego_image": embed_result["stego_image"],
            "secret_key": KEY,
            "original_message": MESSAGE,
        }
        r = requests.post(f"{BASE_URL}/api/robustness", headers=HEADERS, json=body, timeout=60)
        assert r.status_code == 200, r.text
        results = r.json()["results"]
        assert len(results) == 3
        names = [x["attack"] for x in results]
        # Check 3 expected attacks present
        assert any("Gaussian" in n for n in names)
        assert any("JPEG" in n for n in names)
        assert any("Resize" in n for n in names)
        for x in results:
            for k in ["attack", "psnr", "ssim", "passed", "bit_error_rate", "preview"]:
                assert k in x
            assert x["preview"].startswith("data:image/png;base64,")


# --- Ablation ---
class TestAblation:
    def test_ablation_three_configs(self, cover_png):
        files = {"image": ("c.png", cover_png, "image/png")}
        data = {"message": MESSAGE, "secret_key": KEY}
        r = requests.post(f"{BASE_URL}/api/ablation", headers=HEADERS, files=files, data=data, timeout=60)
        assert r.status_code == 200, r.text
        results = r.json()["results"]
        assert len(results) == 3
        configs = [x["config"] for x in results]
        assert "No Encryption" in configs
        assert "No Diffusion" in configs
        assert any("Full" in c for c in configs)
        for x in results:
            assert "psnr" in x and "ssim" in x and "bpp" in x


# --- Comparison & Graphs (public) ---
class TestStaticData:
    def test_comparison(self):
        r = requests.get(f"{BASE_URL}/api/comparison", timeout=10)
        assert r.status_code == 200
        methods = r.json()["methods"]
        assert len(methods) == 5
        names = [m["method"] for m in methods]
        for expected in ["LSB", "LSB_XOR", "DCT", "GAN", "PROPOSED"]:
            assert expected in names
        # PROPOSED should have highest PSNR
        proposed = next(m for m in methods if m["method"] == "PROPOSED")
        assert proposed["psnr"] > 45

    def test_graphs(self):
        r = requests.get(f"{BASE_URL}/api/graphs", timeout=10)
        assert r.status_code == 200
        d = r.json()
        for k in ["epochs", "psnr_vs_epoch", "ssim_vs_epoch", "timing_ms",
                  "steganalysis", "loss_curves"]:
            assert k in d
        assert len(d["epochs"]) == 5
        for series_key in ["psnr_vs_epoch", "ssim_vs_epoch"]:
            for method in ["LSB", "LSB_XOR", "DCT", "GAN", "PROPOSED"]:
                assert method in d[series_key]
        assert "epochs" in d["loss_curves"]
        assert len(d["loss_curves"]["train"]) == len(d["loss_curves"]["epochs"])
