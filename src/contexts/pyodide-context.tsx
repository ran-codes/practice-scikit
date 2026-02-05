"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";
import { runPythonCode, type PythonResult } from "@/lib/run-python";

export type PyodideStatus =
  | "idle"
  | "loading-runtime"
  | "loading-packages"
  | "ready"
  | "error";

interface PyodideContextValue {
  pyodide: any | null;
  status: PyodideStatus;
  error: string | null;
  runPython: (code: string) => Promise<PythonResult>;
  resetNamespace: () => Promise<void>;
}

const PyodideContext = createContext<PyodideContextValue | null>(null);

declare global {
  interface Window {
    loadPyodide: (config: { indexURL: string }) => Promise<any>;
  }
}

export function PyodideProvider({ children }: { children: ReactNode }) {
  const [pyodide, setPyodide] = useState<any>(null);
  const [status, setStatus] = useState<PyodideStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    async function initPyodide() {
      if (typeof window === "undefined") return;

      try {
        setStatus("loading-runtime");

        // Load Pyodide script dynamically
        const script = document.createElement("script");
        script.src =
          "https://cdn.jsdelivr.net/pyodide/v0.27.0/full/pyodide.js";
        script.async = true;

        await new Promise<void>((resolve, reject) => {
          script.onload = () => resolve();
          script.onerror = () => reject(new Error("Failed to load Pyodide"));
          document.head.appendChild(script);
        });

        if (!mounted) return;

        // Initialize Pyodide
        const py = await window.loadPyodide({
          indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.0/full/",
        });

        if (!mounted) return;
        setStatus("loading-packages");

        // Load scikit-learn and dependencies
        await py.loadPackage(["scikit-learn", "pandas", "numpy", "matplotlib", "polars"]);
        await py.runPythonAsync(`
import sklearn
import pandas as pd
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for browser
import matplotlib.pyplot as plt

# Set default figure size for better display
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 100
`);

        if (!mounted) return;
        setPyodide(py);
        setStatus("ready");
      } catch (err) {
        if (!mounted) return;
        setError(err instanceof Error ? err.message : "Unknown error");
        setStatus("error");
      }
    }

    // Small delay to let UI render first
    const timer = setTimeout(initPyodide, 100);

    return () => {
      mounted = false;
      clearTimeout(timer);
    };
  }, []);

  const runPython = useCallback(
    async (code: string): Promise<PythonResult> => {
      if (!pyodide) {
        return { success: false, error: "Pyodide not initialized" };
      }
      return runPythonCode(pyodide, code);
    },
    [pyodide]
  );

  const resetNamespace = useCallback(async () => {
    if (!pyodide) return;

    // Clear all user-defined variables but keep standard imports
    await pyodide.runPythonAsync(`
# Keep only the standard imports and built-ins
_keep_names = {
    # Standard imports we set up
    'sklearn', 'pd', 'np', 'pl', 'plt', 'matplotlib',
    'pandas', 'numpy', 'polars',
    # Built-ins and internals
    '__name__', '__doc__', '__package__', '__loader__', '__spec__',
    '__builtins__', '__file__', '__cached__',
}

# Get all current global names
_all_names = list(globals().keys())

# Delete user-defined variables
for _name in _all_names:
    if _name not in _keep_names and not _name.startswith('_'):
        try:
            del globals()[_name]
        except:
            pass

# Clean up our temp variables
del _keep_names, _all_names, _name
`);
  }, [pyodide]);

  return (
    <PyodideContext.Provider value={{ pyodide, status, error, runPython, resetNamespace }}>
      {children}
    </PyodideContext.Provider>
  );
}

export function usePyodide() {
  const context = useContext(PyodideContext);
  if (!context) {
    throw new Error("usePyodide must be used within PyodideProvider");
  }
  return context;
}
