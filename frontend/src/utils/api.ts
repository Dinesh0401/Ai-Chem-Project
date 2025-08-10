import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:5000"
});

export const generateMolecules = async (z?: number[], n = 4) => {
  const { data } = await api.post("/api/generate", { z, n });
  return data;
};

export const getProps = async (smiles: string[]) => {
  const { data } = await api.post("/api/props", { smiles });
  return data;
};

export const rlOptimize = async (target: any, steps = 20000) => {
  const { data } = await api.post("/api/rl_optimize", { target, steps });
  return data;
};

export const evaluateBatch = async (n = 32) => {
  const { data } = await api.post("/api/evaluate", { n });
  return data;
};

export default api;
