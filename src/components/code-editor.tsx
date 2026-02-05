"use client";

import { useEffect, useRef, useCallback } from "react";
import { EditorState } from "@codemirror/state";
import { EditorView, basicSetup } from "codemirror";
import { python } from "@codemirror/lang-python";

interface Props {
  value: string;
  onChange: (value: string) => void;
}

export function CodeEditor({ value, onChange }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<EditorView | null>(null);
  const isInternalChange = useRef(false);

  const handleChange = useCallback(
    (update: { docChanged: boolean; state: EditorState }) => {
      if (update.docChanged && !isInternalChange.current) {
        onChange(update.state.doc.toString());
      }
    },
    [onChange]
  );

  useEffect(() => {
    if (!containerRef.current) return;

    const state = EditorState.create({
      doc: value,
      extensions: [
        basicSetup,
        python(),
        EditorView.updateListener.of(handleChange),
        EditorView.theme({
          "&": {
            fontSize: "14px",
            height: "100%",
          },
          ".cm-content": {
            padding: "16px 0",
            fontFamily: "var(--font-geist-mono), monospace",
          },
          ".cm-gutters": {
            backgroundColor: "hsl(var(--muted))",
            borderRight: "1px solid hsl(var(--border))",
          },
          ".cm-lineNumbers .cm-gutterElement": {
            padding: "0 8px",
          },
          "&.cm-focused": {
            outline: "none",
          },
          ".cm-scroller": {
            overflow: "auto",
          },
        }),
      ],
    });

    const view = new EditorView({
      state,
      parent: containerRef.current,
    });

    editorRef.current = view;

    return () => {
      view.destroy();
      editorRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run once on mount

  // Sync external value changes
  useEffect(() => {
    const view = editorRef.current;
    if (!view) return;

    const currentDoc = view.state.doc.toString();
    if (currentDoc !== value) {
      isInternalChange.current = true;
      view.dispatch({
        changes: { from: 0, to: currentDoc.length, insert: value },
      });
      isInternalChange.current = false;
    }
  }, [value]);

  return (
    <div
      ref={containerRef}
      className="h-full overflow-hidden bg-background"
    />
  );
}
