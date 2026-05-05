import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
export const API = `${BACKEND_URL}/api`;

export const api = axios.create({
  baseURL: API,
  withCredentials: true,
});

export const auth = {
  me: () => api.get("/auth/me").then((r) => r.data),
  session: (session_id) => api.post("/auth/session", { session_id }).then((r) => r.data),
  logout: () => api.post("/auth/logout").then((r) => r.data),
};

export const sgicApi = {
  embed: (image, message, secret_key) => {
    const fd = new FormData();
    fd.append("image", image);
    fd.append("message", message);
    fd.append("secret_key", secret_key);
    return api.post("/embed", fd).then((r) => r.data);
  },
  extract: (image, secret_key, original_message = "") => {
    const fd = new FormData();
    fd.append("image", image);
    fd.append("secret_key", secret_key);
    fd.append("original_message", original_message);
    return api.post("/extract", fd).then((r) => r.data);
  },
  robustness: (stego_image, secret_key, original_message) =>
    api
      .post("/robustness", { stego_image, secret_key, original_message })
      .then((r) => r.data),
  ablation: (image, message, secret_key) => {
    const fd = new FormData();
    fd.append("image", image);
    fd.append("message", message);
    fd.append("secret_key", secret_key);
    return api.post("/ablation", fd).then((r) => r.data);
  },
  comparison: () => api.get("/comparison").then((r) => r.data),
  graphs: () => api.get("/graphs").then((r) => r.data),
};
