import { useState } from "react";

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
}

interface PaperListComponentProps {
  papers: PaperComponentProps[];
}

export const PaperListComponent = (props: PaperListComponentProps) => {
  const paperCount = props.papers?.length || 0;
  const [selectedPapers, setSelectedPapers] = useState<Set<string>>(new Set());
  
  if (!props.papers || paperCount === 0) {
    return (
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-6 text-center">
        <p className="text-gray-500">No papers found.</p>
      </div>
    );
  }

  const handleSelectChange = (paperId: string, selected: boolean) => {
    setSelectedPapers((prev) => {
      const newSet = new Set(prev);
      if (selected) {
        newSet.add(paperId);
      } else {
        newSet.delete(paperId);
      }
      return newSet;
    });
  };

  const selectedCount = selectedPapers.size;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between bg-gradient-to-r from-emerald-50 to-teal-50 rounded-lg px-4 py-3 border border-emerald-200">
        <h3 className="text-lg font-semibold text-gray-900">
          📚 Found {paperCount} Paper{paperCount !== 1 ? 's' : ''}
        </h3>
        <span className="text-sm text-gray-600">
          {selectedCount > 0 ? `${selectedCount} selected` : 'Select papers to save'}
        </span>
      </div>

      {/* Papers List */}
      <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
        {props.papers.map((paper, index) => (
          <PaperComponent 
            key={paper.paperId || paper.url || index}
            {...paper}
            isSelected={selectedPapers.has(paper.paperId || '')}
            onSelectChange={handleSelectChange}
          />
        ))}
      </div>
    </div>
  );
};

export const PaperComponent = (props: PaperComponentProps) => {
  // Helper to truncate abstract to ~200 words
  const truncateAbstract = (text?: string) => {
    if (!text) return "No abstract available.";
    const words = text.split(/\s+/);
    if (words.length <= 200) return text;
    return words.slice(0, 200).join(" ") + "...";
  };

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
        ? 'border-emerald-400 shadow-md ring-2 ring-emerald-100' 
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
              className="w-5 h-5 rounded border-gray-300 text-emerald-600 focus:ring-emerald-500 focus:ring-2 cursor-pointer"
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
                  className="flex-shrink-0 text-gray-400 hover:text-emerald-600 transition-colors"
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
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-emerald-50 text-emerald-700 text-sm font-medium">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Selected
                </span>
              </div>
            )}

            {/* Abstract */}
            <div className="text-sm text-gray-600 leading-relaxed mb-4">
              {truncateAbstract(props.abstract)}
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
