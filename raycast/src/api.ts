import { getPreferenceValues } from "@raycast/api";

/** Client for the local Tohu.app API (see macos/Sources/TohuApp/APIServer.swift). */

export interface Alternative {
  text: string;
  probability: number;
}

export interface Word {
  start: number;
  end: number;
  /** Offsets into the text in UTF-16 units, i.e. JavaScript string indices. */
  utf16Start: number;
  utf16End: number;
  text: string;
  confidence: number;
  alternatives: Alternative[];
}

export interface Analysis {
  text: string;
  words: Word[];
}

interface Prefs {
  apiUrl?: string;
}

export class TohuError extends Error {}

function baseUrl(): string {
  const { apiUrl } = getPreferenceValues<Prefs>();
  return (apiUrl || "http://127.0.0.1:7744").replace(/\/+$/, "");
}

async function post<T>(path: string, text: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${baseUrl()}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  } catch {
    throw new TohuError("Tohu isn't running. Open Tohu.app first.");
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    throw new TohuError(body.error ?? `Tohu returned HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function restore(text: string): Promise<string> {
  return (await post<{ text: string }>("/restore", text)).text;
}

export function analyze(text: string): Promise<Analysis> {
  return post<Analysis>("/analyze", text);
}
