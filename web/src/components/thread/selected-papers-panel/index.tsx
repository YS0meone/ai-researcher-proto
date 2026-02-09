import { Button } from "@/components/ui/button";
import { usePaperSelection } from "@/providers/PaperSelection";
import { PaperCard } from "./paper-card";
import { X } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { useQueryState, parseAsBoolean } from "nuqs";

function SelectedPapersContent() {
  const { selectedPapers, deselectPaper, clearSelection } = usePaperSelection();
  const [, setSelectedPapersOpen] = useQueryState(
    "selectedPapersOpen",
    parseAsBoolean.withDefault(false)
  );

  const papersArray = Array.from(selectedPapers.values());

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold">My Paper List</h2>
            <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 text-xs font-medium">
              {selectedPapers.size}
            </span>
          </div>
          <p className="text-xs text-gray-500">Papers added to your research collection</p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setSelectedPapersOpen(false)}
          className="h-8 w-8 p-0"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {papersArray.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-8">
            <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mb-4">
              <svg
                className="w-8 h-8 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <h3 className="text-sm font-medium text-gray-900 mb-1">
              No papers in your list
            </h3>
            <p className="text-sm text-gray-500">
              Add papers from search results to build your collection
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {papersArray.map((paper) => (
              <PaperCard
                key={paper.paperId}
                paper={paper}
                onDeselect={deselectPaper}
              />
            ))}
          </div>
        )}
      </div>

      {/* Footer with Clear All button */}
      {papersArray.length > 0 && (
        <div className="p-4 border-t">
          <Button
            variant="outline"
            className="w-full"
            onClick={clearSelection}
          >
            Clear All
          </Button>
        </div>
      )}
    </div>
  );
}

export default function SelectedPapersPanel() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [selectedPapersOpen, setSelectedPapersOpen] = useQueryState(
    "selectedPapersOpen",
    parseAsBoolean.withDefault(false)
  );
  const { selectedPapers } = usePaperSelection();

  // Auto-close mobile sheet when last paper is deselected
  const prevSize = selectedPapers.size;
  if (!isLargeScreen && prevSize > 0 && selectedPapers.size === 0) {
    setSelectedPapersOpen(false);
  }

  return (
    <>
      {/* Desktop: fixed panel */}
      <div className="hidden lg:block h-full">
        <SelectedPapersContent />
      </div>

      {/* Mobile: Sheet overlay */}
      <div className="lg:hidden">
        <Sheet
          open={!!selectedPapersOpen && !isLargeScreen}
          onOpenChange={(open) => {
            if (isLargeScreen) return;
            setSelectedPapersOpen(open);
          }}
        >
          <SheetContent side="right" className="lg:hidden flex w-[85vw] sm:w-[400px]">
            <SheetHeader className="sr-only">
              <SheetTitle>My Paper List</SheetTitle>
            </SheetHeader>
            <SelectedPapersContent />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
