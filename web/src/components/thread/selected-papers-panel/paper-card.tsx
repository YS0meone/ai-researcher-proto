import { X, ExternalLink } from "lucide-react";
import { PaperData } from "@/providers/PaperSelection";

interface PaperCardProps {
  paper: PaperData;
  onDeselect: (paperId: string) => void;
}

export function PaperCard({ paper, onDeselect }: PaperCardProps) {
  // Format first author + "et al."
  const formatAuthors = () => {
    if (!paper.authors || paper.authors.length === 0) {
      return "Unknown authors";
    }

    const firstAuthor = typeof paper.authors[0] === "string"
      ? paper.authors[0]
      : paper.authors[0].name || "Unknown";

    if (paper.authors.length > 1) {
      return `${firstAuthor} et al.`;
    }
    return firstAuthor;
  };

  return (
    <div className="group relative rounded-lg border border-gray-200 bg-white p-3 hover:border-emerald-300 hover:shadow-sm transition-all duration-200">
      {/* Deselect button */}
      <button
        onClick={() => paper.paperId && onDeselect(paper.paperId)}
        className="absolute top-2 right-2 p-1 rounded-md text-gray-400 hover:text-red-600 hover:bg-red-50 transition-colors opacity-0 group-hover:opacity-100"
        aria-label="Deselect paper"
      >
        <X className="w-4 h-4" />
      </button>

      {/* Title - truncated to 2 lines */}
      <h4 className="text-sm font-semibold text-gray-900 mb-2 pr-6 line-clamp-2 leading-tight">
        {paper.title || "Untitled Paper"}
      </h4>

      {/* Author and year */}
      <div className="flex items-center justify-between text-xs text-gray-600 mb-2">
        <span className="truncate flex-1">{formatAuthors()}</span>
        {paper.year && (
          <span className="ml-2 flex-shrink-0">{paper.year}</span>
        )}
      </div>

      {/* Citation count and link */}
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center gap-1 text-gray-500">
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
          </svg>
          <span>{paper.citationCount?.toLocaleString() || 0}</span>
        </div>

        {paper.url && (
          <a
            href={paper.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-emerald-600 transition-colors"
            aria-label="Open paper"
          >
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        )}
      </div>
    </div>
  );
}
