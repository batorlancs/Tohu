import { Clipboard, getSelectedText, showHUD } from "@raycast/api";
import { restore } from "./api";

/** Restore the accents of the selected text and paste the result over the selection. */
export default async function Command() {
  let text: string;
  try {
    text = await getSelectedText();
  } catch {
    await showHUD("No text selected");
    return;
  }
  if (!text.trim()) {
    await showHUD("No text selected");
    return;
  }

  try {
    const fixed = await restore(text);
    if (fixed === text) {
      await showHUD("Already accented");
      return;
    }
    await Clipboard.paste(fixed);
  } catch (e) {
    await showHUD(e instanceof Error ? e.message : String(e));
  }
}
