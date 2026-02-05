import { notFound } from "next/navigation";
import { PHASE_INFO } from "@/data/types";
import { getExercisesByPhase } from "@/data/course";
import { PhasePageClient } from "./phase-page-client";

interface Props {
  params: Promise<{ phase: string }>;
}

export default async function PhasePage({ params }: Props) {
  const { phase: phaseStr } = await params;
  const phase = parseInt(phaseStr) as 1 | 2 | 3;

  if (![1, 2, 3].includes(phase)) {
    notFound();
  }

  const info = PHASE_INFO[phase];
  const exercises = getExercisesByPhase(phase);

  return (
    <PhasePageClient
      phase={phase}
      info={info}
      exercises={exercises}
    />
  );
}

export function generateStaticParams() {
  return [{ phase: "1" }, { phase: "2" }, { phase: "3" }];
}
