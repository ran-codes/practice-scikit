import { notFound } from "next/navigation";
import { puzzles, getPuzzleById } from "@/data/puzzles";
import { PuzzlePage } from "@/components/puzzle-page";

interface Props {
  params: Promise<{ id: string }>;
}

export default async function Page({ params }: Props) {
  const { id } = await params;
  const puzzle = getPuzzleById(id);

  if (!puzzle) {
    notFound();
  }

  return <PuzzlePage puzzle={puzzle} />;
}

export function generateStaticParams() {
  return puzzles.map((puzzle) => ({ id: puzzle.id }));
}
