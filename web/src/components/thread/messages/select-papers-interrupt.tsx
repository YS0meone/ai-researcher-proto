import { CheckCircle } from "lucide-react";
import { usePaperSelection } from "@/providers/PaperSelection";

export function SelectPapersInterruptView() {
  const { selectedPaperIds } = usePaperSelection();
  const count = selectedPaperIds.length;

  return (
    <div className="border border-emerald-200 bg-emerald-50 rounded-lg px-4 py-3 flex items-start gap-3">
      <CheckCircle className="text-emerald-600 mt-0.5 shrink-0 w-5 h-5" />
      <div className="text-sm text-emerald-900">
        <p className="font-medium mb-0.5">Papers retrieved</p>
        <p className="text-emerald-700">
          {count > 0
            ? `${count} paper${count !== 1 ? "s" : ""} selected. `
            : "No papers selected yet. "}
          Select papers from the list above, then type what you'd like to do
          next — or click <strong>Continue</strong> to proceed.
        </p>
      </div>
    </div>
  );
}
