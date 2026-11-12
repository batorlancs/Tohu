import { Action, ActionPanel, Color, Icon, List } from "@raycast/api";
import { useEffect, useMemo, useRef, useState } from "react";
import { analyze, Analysis, Word } from "./api";

/** A word the user picked by hand; kept while that span of the input is unchanged. */
interface Pin {
  start: number;
  end: number;
  input: string;
  form: string;
  original: Word;
}

const THRESHOLD = 0.9;
const pct = (p: number) => `${Math.round(p * 100)}%`;
const confidenceColor = (p: number) => (p < 0.6 ? Color.Red : p < 0.8 ? Color.Orange : Color.Yellow);

/** Escape text for Raycast markdown, keeping line breaks. */
const escapeMd = (s: string) => s.replace(/[\\`*_{}[\]()#+\-.!|<>~]/g, "\\$&").replace(/\n/g, "  \n");

type Mark = "uncertain" | "focused" | "chosen";

/** The output as markdown: uncertain words bold, the focused one boxed, chosen ones italic. */
function renderMarkdown(text: string, marks: { start: number; end: number; mark: Mark }[]): string {
  let md = "";
  let pos = 0;
  for (const m of [...marks].sort((a, b) => a.start - b.start)) {
    if (m.start < pos || m.end > text.length) continue;
    const word = text.slice(m.start, m.end);
    md += escapeMd(text.slice(pos, m.start));
    md += m.mark === "focused" ? `\`${word}\`` : m.mark === "chosen" ? `_${escapeMd(word)}_` : `**${escapeMd(word)}**`;
    pos = m.end;
  }
  return md + escapeMd(text.slice(pos));
}

/**
 * Type without accents in the search bar; the detail pane shows the accented text in full.
 * Uncertain words are listed on the left with their alternatives; ↵ pastes into the frontmost app.
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
      setLoading(false);
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
  const { output, uncertain, chosen } = useMemo(() => {
    let out = result.text;
    const chosen: Pin[] = [];
    for (const p of Object.values(pins)) {
      if (input.slice(p.start, p.end) === p.input && p.end <= out.length) {
        chosen.push(p);
        out = out.slice(0, p.start) + p.form + out.slice(p.end);
      }
    }
    const pinned = new Set(chosen.map((p) => p.start));
    const uncertain = result.words.filter(
      (w) => w.confidence < THRESHOLD && w.alternatives.length > 0 && !pinned.has(w.utf16Start),
    );
    return { output: out, uncertain, chosen: chosen.sort((a, b) => a.start - b.start) };
  }, [result, pins, input]);

  const choose = (w: Word, form: string) =>
    setPins((p) => ({
      ...p,
      [w.utf16Start]: {
        start: w.utf16Start,
        end: w.utf16End,
        input: input.slice(w.utf16Start, w.utf16End),
        form,
        original: w,
      },
    }));
  const unpin = (start: number) =>
    setPins((p) => Object.fromEntries(Object.entries(p).filter(([k]) => Number(k) !== start)));

  const markdown = (focus?: number) =>
    renderMarkdown(output, [
      ...uncertain.map((w) => ({
        start: w.utf16Start,
        end: w.utf16End,
        mark: (w.utf16Start === focus ? "focused" : "uncertain") as Mark,
      })),
      ...chosen.map((p) => ({ start: p.start, end: p.end, mark: (p.start === focus ? "focused" : "chosen") as Mark })),
    ]);

  const outputActions = (
    <ActionPanel.Section>
      <Action.Paste title="Paste Accented Text" content={output} />
      <Action.CopyToClipboard title="Copy Accented Text" content={output} />
      <Action
        title="Clear"
        icon={Icon.Eraser}
        shortcut={{ modifiers: ["cmd", "shift"], key: "backspace" }}
        onAction={() => {
          setInput("");
          setPins({});
        }}
      />
    </ActionPanel.Section>
  );

  const showDetail = !!input && !error;

  return (
    <List
      navigationTitle="Add Accents as You Type"
      searchBarPlaceholder="Írj ékezetek nélkül…  (irj ekezetek nelkul)"
      searchText={input}
      onSearchTextChange={setInput}
      filtering={false}
      throttle={false}
      isLoading={loading}
      isShowingDetail={showDetail}
    >
      {error ? (
        <List.EmptyView icon={{ source: Icon.Plug, tintColor: Color.Red }} title={error} />
      ) : !input ? (
        <List.EmptyView
          icon="icon.png"
          title="Type Hungarian without accents"
          description="The accented text appears as you type · ↵ pastes it into the frontmost app"
        />
      ) : (
        <>
          <List.Section title="Result">
            <List.Item
              id="result"
              icon={{ source: Icon.Text, tintColor: uncertain.length ? Color.Orange : Color.Green }}
              title="Accented text"
              accessories={[
                uncertain.length
                  ? { tag: { value: `${uncertain.length} to check`, color: Color.Orange } }
                  : { icon: { source: Icon.CheckCircle, tintColor: Color.Green } },
              ]}
              detail={<List.Item.Detail markdown={markdown()} />}
              actions={<ActionPanel>{outputActions}</ActionPanel>}
            />
          </List.Section>

          {uncertain.length > 0 && (
            <List.Section title="To check" subtitle={`${uncertain.length}`}>
              {uncertain.map((w) => (
                <List.Item
                  key={`u${w.utf16Start}`}
                  id={`u${w.utf16Start}`}
                  icon={{ source: Icon.QuestionMarkCircle, tintColor: confidenceColor(w.confidence) }}
                  title={w.text}
                  subtitle={w.alternatives.map((a) => a.text).join(", ")}
                  accessories={[{ tag: { value: pct(w.confidence), color: confidenceColor(w.confidence) } }]}
                  detail={
                    <List.Item.Detail
                      markdown={markdown(w.utf16Start)}
                      metadata={
                        <List.Item.Detail.Metadata>
                          <List.Item.Detail.Metadata.TagList title="Tohu picked">
                            <List.Item.Detail.Metadata.TagList.Item
                              text={`${w.text}  ${pct(w.confidence)}`}
                              color={Color.Green}
                              onAction={() => choose(w, w.text)}
                            />
                          </List.Item.Detail.Metadata.TagList>
                          <List.Item.Detail.Metadata.TagList title="Alternatives">
                            {w.alternatives.map((a) => (
                              <List.Item.Detail.Metadata.TagList.Item
                                key={a.text}
                                text={`${a.text}  ${pct(a.probability)}`}
                                color={Color.Orange}
                                onAction={() => choose(w, a.text)}
                              />
                            ))}
                          </List.Item.Detail.Metadata.TagList>
                        </List.Item.Detail.Metadata>
                      }
                    />
                  }
                  actions={
                    <ActionPanel>
                      <ActionPanel.Section title={`“${w.text}”`}>
                        {w.alternatives.map((a, i) => (
                          <Action
                            key={a.text}
                            title={`Use “${a.text}”`}
                            icon={Icon.ArrowRightCircle}
                            shortcut={i < 9 ? { modifiers: ["cmd"], key: `${i + 1}` as "1" } : undefined}
                            onAction={() => choose(w, a.text)}
                          />
                        ))}
                        <Action
                          title={`Keep “${w.text}”`}
                          icon={Icon.CheckCircle}
                          shortcut={{ modifiers: ["cmd"], key: "0" }}
                          onAction={() => choose(w, w.text)}
                        />
                      </ActionPanel.Section>
                      {outputActions}
                    </ActionPanel>
                  }
                />
              ))}
            </List.Section>
          )}

          {chosen.length > 0 && (
            <List.Section title="Chosen">
              {chosen.map((p) => (
                <List.Item
                  key={`c${p.start}`}
                  id={`c${p.start}`}
                  icon={{ source: Icon.CheckCircle, tintColor: Color.Blue }}
                  title={p.form}
                  subtitle={p.form === p.original.text ? "kept" : `instead of ${p.original.text}`}
                  detail={<List.Item.Detail markdown={markdown(p.start)} />}
                  actions={
                    <ActionPanel>
                      <ActionPanel.Section>
                        <Action
                          title="Undo Choice"
                          icon={Icon.Undo}
                          shortcut={{ modifiers: ["cmd"], key: "z" }}
                          onAction={() => unpin(p.start)}
                        />
                      </ActionPanel.Section>
                      {outputActions}
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
