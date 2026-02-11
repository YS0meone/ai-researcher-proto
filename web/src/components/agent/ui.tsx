import { useState } from "react";
import { toast } from "sonner";
import { usePaperSelection, PaperData } from "@/providers/PaperSelection";

// ============================================
// StepTracker Component
// ============================================

interface Step {
  id: string;
  label: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  description?: string;
}

interface StepTrackerProps {
  steps: Step[];
}

export const StepTracker = ({ steps }: StepTrackerProps) => {
  return (
    <div className="w-xl max-w-3xl rounded-lg border border-gray-200 bg-white p-4 space-y-3 shadow-sm">
      <div className="flex items-center gap-2 mb-3">
        <svg className="w-5 h-5 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
        <h3 className="font-semibold text-gray-900">Agent Actions</h3>
      </div>

      <div className="space-y-2">
        {steps.map((step, index) => (
          <div
            key={step.id}
            className={`flex items-start gap-3 p-3 rounded-md transition-all ${
              step.status === 'running' ? 'bg-gray-50 border border-gray-200' : ''
            }`}
          >
            {/* Status Icon */}
            <div className="flex-shrink-0 mt-0.5">
              {step.status === 'completed' && (
                <svg className="w-5 h-5 text-gray-900" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              )}
              {step.status === 'running' && (
                <svg className="w-5 h-5 text-gray-900 animate-spin" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
              )}
              {step.status === 'pending' && (
                <div className="w-5 h-5 rounded-full border-2 border-gray-300" />
              )}
              {step.status === 'error' && (
                <svg className="w-5 h-5 text-red-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              )}
            </div>

            {/* Step Content */}
            <div className="flex-1 min-w-0">
              <div className={`font-medium ${
                step.status === 'running' ? 'text-gray-900' :
                step.status === 'completed' ? 'text-gray-700' :
                step.status === 'error' ? 'text-red-700' :
                'text-gray-500'
              }`}>
                {step.label}
              </div>
              {step.description && (
                <div className="text-sm text-gray-600 mt-1">
                  {step.description}
                </div>
              )}
            </div>

            {/* Connector Line (except for last item) */}
            {index < steps.length - 1 && step.status !== 'pending' && (
              <div className="absolute left-[34px] top-10 w-0.5 h-full bg-gray-200"
                   style={{ height: '24px' }} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

// ============================================
// Paper Components
// ============================================

interface Author {
  authorId?: string;
  name?: string;
}

interface PublicationVenue {
  name?: string;
  type?: string;
}

interface PaperComponentProps {
  paperId?: string;
  title?: string;
  authors?: Author[] | string[];
  venue?: string;
  publicationVenue?: PublicationVenue;
  publicationDate?: string;
  year?: number;
  abstract?: string;
  citationCount?: number;
  url?: string;
  isSelected?: boolean;
  onSelectChange?: (paperId: string, selected: boolean) => void;
  openAccessPdf?: { url?: string } | null;
}

interface PaperListComponentProps {
  papers: PaperComponentProps[];
}

interface IngestStatus {
  status: 'idle' | 'ingesting' | 'success' | 'error';
  message?: string;
  taskIds?: { paperId: string; taskId: string }[];
}

export const PaperListComponent = (props: PaperListComponentProps) => {
  const paperCount = props.papers?.length || 0;
  const { selectPaper } = usePaperSelection(); // Only used to add to persistent list after ingestion
  const [tempSelected, setTempSelected] = useState<Set<string>>(new Set()); // Temporary checkbox selections
  const [ingestStatus, setIngestStatus] = useState<IngestStatus>({ status: 'idle' });
  const backendApiUrl = import.meta.env.VITE_BACKEND_API_URL || 'http://localhost:2024';

  if (!props.papers || paperCount === 0) {
    return (
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-6 text-center">
        <p className="text-gray-500">No papers found.</p>
      </div>
    );
  }

  const handleSelectChange = (paperId: string, selected: boolean) => {
    setTempSelected(prev => {
      const newSet = new Set(prev);
      if (selected) {
        newSet.add(paperId);
      } else {
        newSet.delete(paperId);
      }
      return newSet;
    });
  };

  const handleAddToPaperList = async () => {
    if (tempSelected.size === 0) return;

    setIngestStatus({ status: 'ingesting', message: 'Adding papers to your list...' });

    try {
      // Filter papers to only include selected ones
      const papersToIngest = props.papers.filter(
        (p) => p.paperId && tempSelected.has(p.paperId)
      );

      // Make POST request to backend using backend API URL
      const response = await fetch(`${backendApiUrl}/ingest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          papers: papersToIngest,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to add papers (${response.status}): ${errorText || response.statusText}`);
      }

      const result = await response.json();

      // Add papers to persistent list (shown in side panel)
      papersToIngest.forEach(paper => {
        const paperData: PaperData = {
          paperId: paper.paperId,
          title: paper.title,
          authors: paper.authors,
          venue: paper.venue,
          publicationVenue: paper.publicationVenue,
          publicationDate: paper.publicationDate,
          year: paper.year,
          abstract: paper.abstract,
          citationCount: paper.citationCount,
          url: paper.url,
          openAccessPdf: paper.openAccessPdf,
        };
        selectPaper(paperData);
      });

      setIngestStatus({
        status: 'success',
        message: `Successfully added ${tempSelected.size} paper(s) to your list!`,
        taskIds: result.tasks,
      });

      // Show success toast with info about ingestion
      const newlyIngested = result.tasks?.filter((t: any) => t.status === 'queued').length || 0;
      toast.success('Papers Added to Your List', {
        description: newlyIngested > 0
          ? `${tempSelected.size} paper(s) added. ${newlyIngested} new paper(s) are being processed in the background.`
          : `${tempSelected.size} paper(s) added to your list.`,
      });

      // Clear temporary selections
      setTempSelected(new Set());
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to add papers';
      setIngestStatus({
        status: 'error',
        message: errorMessage,
      });

      // Show error toast
      toast.error('Failed to Add Papers', {
        description: errorMessage,
      });
    }
  };

  const selectedCount = tempSelected.size;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between bg-white rounded-lg px-4 py-3 border border-gray-200 shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900">
          📚 Found {paperCount} Paper{paperCount !== 1 ? 's' : ''}
        </h3>
        <span className="text-sm text-gray-600">
          {selectedCount > 0 ? `${selectedCount} selected` : 'Select papers to add to your list'}
        </span>
      </div>

      {/* Action Bar with Add Button */}
      {selectedCount > 0 && (
        <div className="flex flex-col gap-2 bg-white rounded-lg px-4 py-3 border border-gray-200 shadow-sm">
          <div className="flex items-center gap-3">
            <button
              onClick={handleAddToPaperList}
              disabled={ingestStatus.status === 'ingesting'}
              className={`px-4 py-2 rounded-md font-medium text-white transition-all duration-200 ${
                ingestStatus.status === 'ingesting'
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-gray-900 hover:bg-black active:scale-95'
              }`}
            >
              {ingestStatus.status === 'ingesting' ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Adding...
                </span>
              ) : (
                `Add ${selectedCount} to Paper List`
              )}
            </button>

            {ingestStatus.message && (
              <span className={`text-sm ${
                ingestStatus.status === 'success' ? 'text-gray-900' :
                ingestStatus.status === 'error' ? 'text-red-600' :
                'text-gray-600'
              }`}>
                {ingestStatus.message}
              </span>
            )}
          </div>

          <p className="text-xs text-gray-500">
            New papers will be processed and added to your database in the background
          </p>
        </div>
      )}

      {/* Papers List */}
      <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
        {props.papers.map((paper, index) => (
          <PaperComponent
            key={paper.paperId || paper.url || index}
            {...paper}
            isSelected={paper.paperId ? tempSelected.has(paper.paperId) : false}
            onSelectChange={handleSelectChange}
          />
        ))}
      </div>
    </div>
  );
};

export const PaperComponent = (props: PaperComponentProps) => {
  const [isAbstractExpanded, setIsAbstractExpanded] = useState(false);

  // Helper to format authors
  const formatAuthors = () => {
    if (!props.authors || props.authors.length === 0) {
      return "Unknown authors";
    }
    
    // Handle array of author objects or strings
    const authorNames = props.authors.map((author: Author | string) => {
      if (typeof author === "string") return author;
      return author.name || "Unknown";
    });
    
    if (authorNames.length > 4) {
      return authorNames.slice(0, 4).join(", ") + ` ... ${authorNames[authorNames.length - 1]}`;
    }
    return authorNames.join(", ");
  };

  // Get venue name (prefer publicationVenue.name, fallback to venue)
  const venueName = props.publicationVenue?.name || props.venue || "Unknown venue";

  const handleCheckboxChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (props.onSelectChange && props.paperId) {
      props.onSelectChange(props.paperId, e.target.checked);
    }
  };

  return (
    <div className={`rounded-lg border transition-all duration-200 bg-white ${
      props.isSelected
        ? 'border-gray-900 shadow-md ring-2 ring-gray-200'
        : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
    }`}>
      <div className="p-5">
        <div className="flex gap-4">
          {/* Checkbox */}
          <div className="flex-shrink-0 pt-1">
            <input
              type="checkbox"
              checked={props.isSelected || false}
              onChange={handleCheckboxChange}
              className="w-5 h-5 rounded border-gray-300 text-gray-900 focus:ring-gray-900 focus:ring-2 cursor-pointer"
            />
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            {/* Title with external link */}
            <div className="flex items-start gap-2 mb-3">
              <h3 className="text-lg font-semibold text-gray-900 flex-1 leading-tight">
                {props.title || "Untitled Paper"}
              </h3>
              {props.url && (
                <a
                  href={props.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex-shrink-0 text-gray-400 hover:text-gray-900 transition-colors"
                  aria-label="Open paper"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
              )}
            </div>

            {/* Authors */}
            <div className="text-sm text-gray-700 mb-2">
              {formatAuthors()}
            </div>

            {/* Venue and Year */}
            <div className="flex items-center gap-2 text-sm text-gray-600 mb-3">
              <span>{venueName}</span>
              {props.year && (
                <>
                  <span className="text-gray-400">•</span>
                  <span>{props.year}</span>
                </>
              )}
            </div>

            {/* Relevant badge (if selected) */}
            {props.isSelected && (
              <div className="mb-3">
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-gray-900 text-white text-sm font-medium">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Selected
                </span>
              </div>
            )}

            {/* Abstract - Click to expand/collapse */}
            <div
              className="mb-4 cursor-pointer group"
              onClick={() => setIsAbstractExpanded(!isAbstractExpanded)}
            >
              <div className={`text-sm text-gray-600 leading-relaxed ${
                isAbstractExpanded ? '' : 'line-clamp-3'
              }`}>
                {props.abstract || "No abstract available."}
              </div>
              {props.abstract && props.abstract.length > 200 && (
                <button className="text-xs text-emerald-600 hover:text-emerald-700 mt-1 font-medium">
                  {isAbstractExpanded ? '− Show less' : '+ Show more'}
                </button>
              )}
            </div>

            {/* Footer: Citation count and actions */}
            <div className="flex items-center justify-between pt-3 border-t border-gray-100">
              {/* Citation count */}
              <div className="flex items-center gap-2 text-sm">
                <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                </svg>
                <span className="text-gray-600">
                  Cited by <span className="font-semibold text-gray-900">{props.citationCount?.toLocaleString() || 0}</span>
                </span>
              </div>

              {/* Action buttons */}
              <div className="flex items-center gap-2">
                <button
                  className="p-2 rounded-md hover:bg-gray-100 text-gray-500 hover:text-emerald-600 transition-colors"
                  aria-label="Like"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                  </svg>
                </button>
                <button
                  className="p-2 rounded-md hover:bg-gray-100 text-gray-500 hover:text-red-600 transition-colors"
                  aria-label="Dislike"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14H5.236a2 2 0 01-1.789-2.894l3.5-7A2 2 0 018.736 3h4.018a2 2 0 01.485.06l3.76.94m-7 10v5a2 2 0 002 2h.096c.5 0 .905-.405.905-.904 0-.715.211-1.413.608-2.008L17 13V4m-7 10h2m5-10h2a2 2 0 012 2v6a2 2 0 01-2 2h-2.5" />
                  </svg>
                </button>
                <button
                  className="p-2 rounded-md hover:bg-gray-100 text-gray-500 hover:text-blue-600 transition-colors"
                  aria-label="Bookmark"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
