import { TriangleAlert, CheckCircle2 } from "lucide-react";
import { usePaperSelection } from "@/providers/PaperSelection";
import { useStreamContext } from "@/providers/Stream";
import { Button } from "@/components/ui/button";

export function SelectPapersInterruptView() {
  const { selectedPaperIds } = usePaperSelection();
  const stream = useStreamContext();
  const count = selectedPaperIds.length;
  const hasSelection = count > 0;

  const handleYes = () => {
    stream.submit(
      {},
      {
        command: {
          resume: {
            selected_paper_ids: selectedPaperIds,
            user_message: null,
          },
        },
        streamMode: ["values"],
      },
    );
  };

  const handleNo = () => {
    stream.submit(
      {},
      {
        command: {
          resume: {
            selected_paper_ids: [],
            user_message: null,
          },
        },
        streamMode: ["values"],
      },
    );
  };

  return (
    <div
      className={`rounded-lg border-l-4 px-4 py-4 flex flex-col gap-3 transition-colors ${
        hasSelection
          ? "border-l-emerald-500 border border-emerald-200 bg-emerald-50"
          : "border-l-amber-500 border border-amber-200 bg-amber-50"
      }`}
    >
      <div className="flex items-start gap-3">
        {hasSelection ? (
          <CheckCircle2 className="text-emerald-600 mt-0.5 shrink-0 w-5 h-5" />
        ) : (
          <TriangleAlert className="text-amber-500 mt-0.5 shrink-0 w-5 h-5" />
        )}
        <div className="text-sm">
          <p className={`font-semibold mb-0.5 ${hasSelection ? "text-emerald-900" : "text-amber-900"}`}>
            {hasSelection ? `${count} paper${count !== 1 ? "s" : ""} selected` : "Action required before proceeding"}
          </p>
          <p className={hasSelection ? "text-emerald-700" : "text-amber-800"}>
            {hasSelection
              ? `Your question will be answered using the selected paper${count !== 1 ? "s" : ""}. Make sure ingestion has finished before clicking Proceed.`
              : "Since you asked a question about these papers, their full content must be ingested before an answer can be generated. Select the relevant papers above, click \"Add to list\", and wait for ingestion to complete — then click Proceed. If none of the papers are relevant, click Reject."}
          </p>
        </div>
      </div>
      <div className="flex gap-2 ml-8">
        <Button size="default" onClick={handleYes} disabled={!hasSelection}>
          Proceed
        </Button>
        <Button size="default" variant="outline" onClick={handleNo}>
          Reject
        </Button>
      </div>
    </div>
  );
}
