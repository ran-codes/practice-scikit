"use client";

import { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { Play, RotateCcw } from "lucide-react";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { usePyodide } from "@/contexts/pyodide-context";
import { OutputPanel } from "@/components/output-panel";
import type { Puzzle } from "@/data/puzzles";

// Dynamic import for CodeMirror (client-only)
const CodeEditor = dynamic(
  () => import("@/components/code-editor").then((m) => m.CodeEditor),
  {
    ssr: false,
    loading: () => <Skeleton className="h-full w-full rounded-lg" />,
  }
);

interface Props {
  puzzle: Puzzle;
}

export function PuzzlePage({ puzzle }: Props) {
  const { runPython, status } = usePyodide();
  const [code, setCode] = useState(puzzle.code);
  const [output, setOutput] = useState<{
    html?: string;
    error?: string;
  } | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [activeTab, setActiveTab] = useState("problem");

  const isReady = status === "ready";
  const isLoading = status === "loading-runtime" || status === "loading-sklearn";

  const handleRun = useCallback(async () => {
    if (!isReady || isRunning) return;

    setIsRunning(true);
    setOutput(null);
    setActiveTab("output"); // Switch to output tab when running

    const result = await runPython(code);
    setOutput(result.success ? { html: result.html } : { error: result.error });

    setIsRunning(false);
  }, [isReady, isRunning, runPython, code]);

  const handleReset = useCallback(() => {
    setCode(puzzle.code);
    setOutput(null);
    setActiveTab("problem");
  }, [puzzle.code]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+Enter or Cmd+Enter to run
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handleRun();
      }
      // Ctrl+Shift+R or Cmd+Shift+R to reset
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === "R") {
        e.preventDefault();
        handleReset();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleRun, handleReset]);

  return (
    <div className="h-[calc(100vh-2rem)] p-4">
      <ResizablePanelGroup direction="horizontal" className="h-full rounded-lg border">
        {/* Left Panel - Code Editor */}
        <ResizablePanel defaultSize={55} minSize={30}>
          <div className="flex flex-col h-full">
            {/* Editor Toolbar */}
            <div className="flex items-center justify-between p-3 border-b bg-muted/30">
              <div className="flex items-center gap-3">
                <h1 className="font-semibold truncate">{puzzle.title}</h1>
                {puzzle.difficulty && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground capitalize">
                    {puzzle.difficulty}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      onClick={handleRun}
                      disabled={!isReady || isRunning}
                    >
                      <Play className="h-4 w-4 mr-1" />
                      {isRunning ? "Running..." : isLoading ? "Loading..." : "Run"}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Run code (Ctrl+Enter)</p>
                  </TooltipContent>
                </Tooltip>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handleReset}
                      disabled={isRunning}
                    >
                      <RotateCcw className="h-4 w-4 mr-1" />
                      Reset
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Reset code (Ctrl+Shift+R)</p>
                  </TooltipContent>
                </Tooltip>
              </div>
            </div>

            {/* Editor */}
            <div className="flex-1 overflow-hidden">
              <CodeEditor value={code} onChange={setCode} />
            </div>
          </div>
        </ResizablePanel>

        <ResizableHandle withHandle />

        {/* Right Panel - Problem & Output */}
        <ResizablePanel defaultSize={45} minSize={25}>
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="h-full flex flex-col"
          >
            <div className="border-b bg-muted/30">
              <TabsList className="w-full justify-start rounded-none border-none bg-transparent h-auto p-0">
                <TabsTrigger
                  value="problem"
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent px-4 py-3"
                >
                  Problem
                </TabsTrigger>
                <TabsTrigger
                  value="output"
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-primary data-[state=active]:bg-transparent px-4 py-3"
                >
                  Output
                  {output && (
                    <span
                      className={`ml-2 w-2 h-2 rounded-full ${
                        output.error ? "bg-destructive" : "bg-green-500"
                      }`}
                    />
                  )}
                </TabsTrigger>
              </TabsList>
            </div>

            <TabsContent value="problem" className="flex-1 overflow-auto m-0 p-4">
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <h2 className="text-lg font-semibold mb-2">Task</h2>
                <p className="text-muted-foreground">{puzzle.description}</p>

                {puzzle.context && (
                  <>
                    <h3 className="text-md font-semibold mt-6 mb-2">Why It Matters</h3>
                    <p className="text-sm text-muted-foreground bg-muted/50 p-3 rounded-md border-l-2 border-primary">
                      {puzzle.context}
                    </p>
                  </>
                )}

                <h3 className="text-md font-semibold mt-6 mb-2">Instructions</h3>
                <ol className="list-decimal list-inside space-y-1 text-sm text-muted-foreground">
                  <li>Examine the setup code and data</li>
                  <li>Modify the code to solve the puzzle</li>
                  <li>Click Run or press Ctrl+Enter to execute</li>
                </ol>
              </div>
            </TabsContent>

            <TabsContent value="output" className="flex-1 overflow-auto m-0 p-4">
              <OutputPanel output={output} isRunning={isRunning} />
            </TabsContent>
          </Tabs>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  );
}
