import { Action, ActionPanel, Clipboard, closeMainWindow, Icon, List, showHUD } from "@raycast/api";
import { useEffect, useMemo, useRef, useState } from "react";
import { analyze, Analysis, Word } from "./api";

/** A word the user picked by hand; kept while that span of the input is unchanged. */
interface Pin {
  end: number;
  input: string;
  form: string;
}

const THRESHOLD = 0.9;
const pct = (p: number) => `${Math.round(p * 100)}%`;

/**
 * Type without accents in the search bar; the first item is the accented text (↵ pastes it into
 * the frontmost app). Uncertain words are listed below with their alternatives.
 */
export default function Command() {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<Analysis>({ text: "", words: [] });
  const [error, setError] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [pins, setPins] = useState<Record<number, Pin>>({});
  const seq = useRef(0);

  useEffect(() => {
    const id = ++seq.current;
    if (!input) {
      setResult({ text: "", words: [] });
      return;
    }
    setLoading(true);
    const timer = setTimeout(async () => {
      try {
        const a = await analyze(input);
        if (id === seq.current) {
          setResult(a);
          setError(undefined);
        }
      } catch (e) {
        if (id === seq.current) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (id === seq.current) setLoading(false);
      }
    }, 60);
    return () => clearTimeout(timer);
  }, [input]);

  // Restoration never changes length, so output offsets are input offsets.
  const { output, uncertain } = useMemo(() => {
    let out = result.text;
    const live: Record<number, Pin> = {};
    for (const [k, p] of Object.entries(pins)) {
      const start = Number(k);
      if (input.slice(start, p.end) === p.input && p.end <= out.length) {
        live[start] = p;
        out = out.slice(0, start) + p.form + out.slice(p.end);
      }
    }
    const uncertain = result.words.filter(
      (w) => w.confidence < THRESHOLD && w.alternatives.length > 0 && !live[w.utf16Start],
    );
    return { output: out, uncertain };
  }, [result, pins, input]);

  const choose = (w: Word, form: string) =>
    setPins((p) => ({ ...p, [w.utf16Start]: { end: w.utf16End, input: input.slice(w.utf16Start, w.utf16End), form } }));

  const paste = async () => {
    await closeMainWindow();
    await Clipboard.paste(output);
  };
  const copy = async () => {
    await Clipboard.copy(output);
    await showHUD("Copied");
  };

  const mainActions = (
    <>
      <Action title="Paste to Frontmost App" icon={Icon.Clipboard} onAction={paste} />
      <Action
        title="Copy to Clipboard"
        icon={Icon.CopyClipboard}
        onAction={copy}
        shortcut={{ modifiers: ["cmd"], key: "c" }}
      />
    </>
  );

  return (
    <List
      searchBarPlaceholder="irj ekezetek nelkul…"
      searchText={input}
      onSearchTextChange={setInput}
      filtering={false}
      isLoading={loading}
      throttle={false}
    >
      {error ? (
        <List.EmptyView icon={Icon.ExclamationMark} title={error} />
      ) : !input ? (
        <List.EmptyView
          icon={Icon.Text}
          title="Type Hungarian without accents"
          description="↵ pastes the accented text"
        />
      ) : (
        <>
          <List.Section title="Result">
            <List.Item
              icon={Icon.Text}
              title={output || input}
              accessories={uncertain.length ? [{ text: `${uncertain.length} uncertain` }] : []}
              actions={<ActionPanel>{mainActions}</ActionPanel>}
            />
          </List.Section>
          {uncertain.length > 0 && (
            <List.Section title="Uncertain words">
              {uncertain.map((w) => (
                <List.Item
                  key={w.utf16Start}
                  icon={Icon.QuestionMark}
                  title={w.text}
                  subtitle={w.alternatives.map((a) => `${a.text} ${pct(a.probability)}`).join(" · ")}
                  accessories={[{ text: pct(w.confidence) }]}
                  actions={
                    <ActionPanel>
                      {w.alternatives.map((a) => (
                        <Action
                          key={a.text}
                          title={`Use “${a.text}”`}
                          icon={Icon.Checkmark}
                          onAction={() => choose(w, a.text)}
                        />
                      ))}
                      <Action title={`Keep “${w.text}”`} icon={Icon.Pin} onAction={() => choose(w, w.text)} />
                      <ActionPanel.Section>{mainActions}</ActionPanel.Section>
                    </ActionPanel>
                  }
                />
              ))}
            </List.Section>
          )}
        </>
      )}
    </List>
  );
}
