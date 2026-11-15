import { DRUG_INFO, fuzzyMatchDrug } from "./drugDatabase";

export type DrugMatch = { name: string; distance: number };
export type OcrEngine = "ai" | "offline";
export type RecognitionResult = { text: string; engine: OcrEngine };

// Optional handwriting reader (Google Gemini). Put the key in .env as EXPO_PUBLIC_GEMINI_API_KEY.
// NOTE: EXPO_PUBLIC_* values are embedded in the app bundle, so anyone can see this key. Demo use only.
const GEMINI_KEY = process.env.EXPO_PUBLIC_GEMINI_API_KEY;
const GEMINI_MODEL = process.env.EXPO_PUBLIC_GEMINI_MODEL || "gemini-2.5-flash";
export const hasHandwritingReader = Boolean(GEMINI_KEY);

const GEMINI_PROMPT =
  "You are reading a photo of a medical prescription (printed or handwritten). " +
  "List every medicine you can read, one per line, as: Name strength (for example: Metformin 500mg). " +
  "Copy the medicine name as written. Do not guess, and do not add medicines that are not visible. " +
  "No headings, no dosage instructions, no commentary. If no medicine is readable, reply with exactly NONE.";

async function imageToBase64(uri: string): Promise<{ data: string; mime: string }> {
  const response = await fetch(uri);
  const blob = await response.blob();
  const dataUrl: string = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Could not read the image file."));
    reader.readAsDataURL(blob);
  });
  const [head, data] = dataUrl.split(",");
  const found = /data:([^;]+)/.exec(head)?.[1] ?? blob.type;
  return { data, mime: found && found.startsWith("image/") ? found : "image/jpeg" };
}

async function recognizeWithGemini(
  imageUri: string,
  onProgress?: (progress: number, status: string) => void,
): Promise<string> {
  onProgress?.(10, "Preparing image...");
  const { data, mime } = await imageToBase64(imageUri);
  onProgress?.(35, "Reading with AI (image is sent to Google Gemini)...");
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": GEMINI_KEY as string },
      body: JSON.stringify({
        contents: [{ parts: [{ text: GEMINI_PROMPT }, { inline_data: { mime_type: mime, data } }] }],
        generationConfig: { temperature: 0 },
      }),
    },
  );
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`AI reader error ${response.status}: ${detail.slice(0, 160)}`);
  }
  const json = await response.json();
  const parts: { text?: string }[] = json?.candidates?.[0]?.content?.parts ?? [];
  const text = parts.map((part) => part.text ?? "").join("\n").trim();
  onProgress?.(90, "Reading finished");
  return /^NONE\.?$/i.test(text) ? "" : text;
}

export async function recognizePrescriptionImage(
  imageUri: string,
  platform: "web" | "ios" | "android",
  onProgress?: (progress: number, status: string) => void,
): Promise<RecognitionResult> {
  // 1) AI reader first (handles handwriting). If it fails or finds nothing, fall back to offline OCR.
  if (hasHandwritingReader) {
    try {
      const aiText = await recognizeWithGemini(imageUri, onProgress);
      if (aiText.trim()) return { text: aiText, engine: "ai" };
    } catch (error) {
      console.warn("[OCR] AI reader failed, using offline OCR instead:", error);
      onProgress?.(10, "AI reader unavailable, using offline OCR...");
    }
  }
  return { text: await recognizeOffline(imageUri, platform, onProgress), engine: "offline" };
}

async function recognizeOffline(
  imageUri: string,
  platform: "web" | "ios" | "android",
  onProgress?: (progress: number, status: string) => void,
): Promise<string> {
  if (platform === "web") {
    const { createWorker } = await import("tesseract.js");
    const worker = await createWorker("eng", 1, {
      logger: (message: { status?: string; progress?: number }) => {
        if (message.status) {
          onProgress?.(Math.round((message.progress ?? 0) * 100), message.status);
        }
      },
    });
    try {
      const { data } = await worker.recognize(imageUri);
      return data.text;
    } finally {
      await worker.terminate();
    }
  }

  try {
    const { default: TextRecognition } = await import(
      "@react-native-ml-kit/text-recognition"
    );
    onProgress?.(25, "Reading text on this device...");
    const result = await TextRecognition.recognize(imageUri);
    onProgress?.(100, "Text recognition complete");
    return result.text;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (/native module|cannot find|not found|undefined|null/i.test(message)) {
      throw new Error(
        "On-device OCR needs an installed iOS/Android development build. Expo Go does not include the text-recognition module. You can still add medicines manually.",
      );
    }
    throw new Error(`Text recognition failed: ${message}`);
  }
}

export function splitRecognizedLines(text: string): string[] {
  const seen = new Set<string>();
  return text
    .split(/\r?\n/)
    .map((line) => line.replace(/^\s*[•●▪*-]+\s*/, "").replace(/\s+/g, " ").trim())
    .filter((line) => line.length >= 3)
    .filter((line) => !/^\s*(prescription|patient name|date|doctor|clinic|hospital|medicines?)\s*[:#-]?\s*$/i.test(line))
    .filter((line) => {
      const key = line.toLowerCase().replace(/[^a-z0-9]/g, "");
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

const IGNORE_PREFIXES = new Set([
  "tab", "tablet", "tabs", "cap", "capsule", "caps", "syrup", "syp",
  "inj", "injection", "take", "give", "use", "medication", "medicine",
  "tb", "tablets", "rx", "sig", "po", "oral", "drops", "oint", "cream",
]);

// Common Indian brand -> generic (single-ingredient brands only; verify before shipping).
// Keys must be lowercase, one word.
export const BRAND_TO_GENERIC: Record<string, string> = {
  dolo: "Paracetamol", crocin: "Paracetamol", calpol: "Paracetamol",
  glycomet: "Metformin", obimet: "Metformin",
  pan: "Pantoprazole", pantocid: "Pantoprazole", pantop: "Pantoprazole",
  ecosprin: "Aspirin", disprin: "Aspirin",
  amlokind: "Amlodipine", amlong: "Amlodipine",
  atorva: "Atorvastatin", storvas: "Atorvastatin", lipitor: "Atorvastatin",
  rosuvas: "Rosuvastatin", crestor: "Rosuvastatin",
  losar: "Losartan", repace: "Losartan",
  metolar: "Metoprolol", betaloc: "Metoprolol",
  cardace: "Ramipril",
  ciplox: "Ciprofloxacin",
  azithral: "Azithromycin", azee: "Azithromycin",
  voveran: "Diclofenac",
  lasix: "Furosemide",
  aldactone: "Spironolactone",
  eltroxin: "Levothyroxine", thyronorm: "Levothyroxine",
  wysolone: "Prednisolone",
  montair: "Montelukast",
  clopitab: "Clopidogrel", plavix: "Clopidogrel",
  januvia: "Sitagliptin", jardiance: "Empagliflozin",
  amaryl: "Glimepiride", diamicron: "Gliclazide",
  coumadin: "Warfarin", lanoxin: "Digoxin",
  zoloft: "Sertraline", prozac: "Fluoxetine", xanax: "Alprazolam",
  tegretol: "Carbamazepine",
};

export function findMedicationMatch(line: string): DrugMatch | null {
  const normalized = line.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  if (!normalized) return null;

  // 1) exact generic name anywhere in the line
  const exact = Object.keys(DRUG_INFO)
    .sort((a, b) => b.length - a.length)
    .find((name) => (` ${normalized} `).includes(` ${name.toLowerCase()} `));
  if (exact) return { name: exact, distance: 0 };

  const words = normalized.split(/\s+/);

  // 2) known brand name (short brands like "pan" need a dose number in the line to avoid false hits)
  const hasNumber = /\d/.test(normalized);
  for (const word of words) {
    const generic = BRAND_TO_GENERIC[word];
    if (generic && DRUG_INFO[generic] && (word.length > 4 || hasNumber)) {
      return { name: generic, distance: 0 };
    }
  }

  // 3) fuzzy match on EVERY alphabetic word (not just the first few), so "Tb Metfrmin 5OOmg" still works
  let best: DrugMatch | null = null;
  for (const rawWord of words) {
    // must start with a letter (skips doses like "5oomg"); fix common OCR digit/letter swaps (0->o, 1->l)
    if (IGNORE_PREFIXES.has(rawWord) || !/^[a-z][a-z0-9]{3,}$/.test(rawWord)) continue;
    const word = rawWord.replace(/0/g, "o").replace(/1/g, "l");
    if (!/^[a-z]{4,}$/.test(word)) continue;
    for (const match of fuzzyMatchDrug(word, 3)) {
      const threshold = Math.max(1, Math.min(3, Math.floor(match.name.length * 0.24)));
      if (match.distance <= threshold && (!best || match.distance < best.distance)) {
        best = match;
      }
    }
  }
  return best;
}

export function extractDosage(line: string): string {
  const match = line.match(/\b\d+(?:\.\d+)?\s?(?:mg|mcg|μg|ug|g|ml|mL|units?|iu|%)\b/i);
  return match?.[0]?.replace(/\s+/g, " ").trim() ?? "As prescribed";
}
