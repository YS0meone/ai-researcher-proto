import { CheckCircle } from "lucide-react";
import { usePaperSelection } from "@/providers/PaperSelection";
import { useStreamContext } from "@/providers/Stream";
import { Button } from "@/components/ui/button";

export function SelectPapersInterruptView({ onNo }: { onNo: () => void }) {
  const { selectedPaperIds } = usePaperSelection();
  const stream = useStreamContext();
  const count = selectedPaperIds.length;

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

  return (
    <div className="border border-emerald-200 bg-emerald-50 rounded-lg px-4 py-4 flex flex-col gap-3">
      <div className="flex items-start gap-3">
        <CheckCircle className="text-emerald-600 mt-0.5 shrink-0 w-5 h-5" />
        <div className="text-sm text-emerald-900">
          <p className="font-medium mb-0.5">Papers retrieved</p>
          <p className="text-emerald-700">
            {count > 0
              ? `${count} paper${count !== 1 ? "s" : ""} selected.`
              : "No papers selected yet."}{" "}
            Would you like to proceed with these results?
          </p>
        </div>
      </div>
      <div className="flex gap-2 ml-8">
        <Button size="sm" onClick={handleYes} disabled={count === 0}>
          Yes, proceed
        </Button>
        <Button size="sm" variant="outline" onClick={onNo}>
          No, search again
        </Button>
      </div>
    </div>
  );
}
