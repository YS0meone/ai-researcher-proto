import {
  createContext,
  useContext,
  ReactNode,
  useCallback,
  useState,
} from "react";

// Paper data structure matching PaperComponentProps from agent/ui.tsx
export interface PaperData {
  paperId?: string;
  title?: string;
  authors?: Array<{ authorId?: string; name?: string }> | string[];
  venue?: string;
  publicationVenue?: { name?: string; type?: string };
  publicationDate?: string;
  year?: number;
  abstract?: string;
  citationCount?: number;
  url?: string;
  openAccessPdf?: { url?: string } | null;
}

interface PaperSelectionContextType {
  selectedPapers: Map<string, PaperData>;
  selectedPaperIds: string[];
  hasSelection: boolean;
  selectPaper: (paper: PaperData) => void;
  deselectPaper: (paperId: string) => void;
  togglePaper: (paper: PaperData) => void;
  clearSelection: () => void;
  isSelected: (paperId: string) => boolean;
}

const PaperSelectionContext = createContext<
  PaperSelectionContextType | undefined
>(undefined);

export function PaperSelectionProvider({ children }: { children: ReactNode }) {
  const [selectedPapers, setSelectedPapers] = useState<Map<string, PaperData>>(
    new Map()
  );

  const selectPaper = useCallback((paper: PaperData) => {
    if (!paper.paperId) return;
    setSelectedPapers((prev) => {
      const newMap = new Map(prev);
      newMap.set(paper.paperId!, paper);
      return newMap;
    });
  }, []);

  const deselectPaper = useCallback((paperId: string) => {
    setSelectedPapers((prev) => {
      const newMap = new Map(prev);
      newMap.delete(paperId);
      return newMap;
    });
  }, []);

  const togglePaper = useCallback((paper: PaperData) => {
    if (!paper.paperId) return;
    setSelectedPapers((prev) => {
      const newMap = new Map(prev);
      if (newMap.has(paper.paperId!)) {
        newMap.delete(paper.paperId!);
      } else {
        newMap.set(paper.paperId!, paper);
      }
      return newMap;
    });
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedPapers(new Map());
  }, []);

  const isSelected = useCallback(
    (paperId: string) => {
      return selectedPapers.has(paperId);
    },
    [selectedPapers]
  );

  const selectedPaperIds = Array.from(selectedPapers.keys());
  const hasSelection = selectedPapers.size > 0;

  const value = {
    selectedPapers,
    selectedPaperIds,
    hasSelection,
    selectPaper,
    deselectPaper,
    togglePaper,
    clearSelection,
    isSelected,
  };

  return (
    <PaperSelectionContext.Provider value={value}>
      {children}
    </PaperSelectionContext.Provider>
  );
}

export function usePaperSelection() {
  const context = useContext(PaperSelectionContext);
  if (context === undefined) {
    throw new Error(
      "usePaperSelection must be used within a PaperSelectionProvider"
    );
  }
  return context;
}
