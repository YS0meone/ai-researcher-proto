interface Author {
  authorId?: string;
  name?: string;
}

interface PublicationVenue {
  name?: string;
  type?: string;
}

interface PaperComponentProps {
  title?: string;
  authors?: Author[] | string[];
  venue?: string;
  publicationVenue?: PublicationVenue;
  publicationDate?: string;
  year?: number;
  abstract?: string;
  citationCount?: number;
  url?: string;
}

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
    
    if (authorNames.length > 5) {
      return authorNames.slice(0, 5).join(", ") + `, et al. (${authorNames.length} authors)`;
    }
    return authorNames.join(", ");
  };

  // Get venue name (prefer publicationVenue.name, fallback to venue)
  const venueName = props.publicationVenue?.name || props.venue || "Unknown venue";
  
  // Format publication date
  const formatDate = () => {
    if (props.publicationDate) {
      try {
        const date = new Date(props.publicationDate);
        return date.toLocaleDateString("en-US", { 
          year: "numeric", 
          month: "long", 
          day: "numeric" 
        });
      } catch {
        return props.year ? `${props.year}` : "Date unknown";
      }
    }
    return props.year ? `${props.year}` : "Date unknown";
  };

  return (
    <div className="rounded-lg border border-gray-200 bg-white shadow-sm hover:shadow-md transition-shadow duration-200 overflow-hidden">
      {/* Header with title */}
      <div className="bg-gradient-to-r from-indigo-50 to-purple-50 px-6 py-4 border-b border-gray-200">
        {props.url ? (
          <a 
            href={props.url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-xl font-semibold text-gray-900 hover:text-indigo-600 transition-colors duration-150 line-clamp-2"
          >
            {props.title || "Untitled Paper"}
          </a>
        ) : (
          <h3 className="text-xl font-semibold text-gray-900 line-clamp-2">
            {props.title || "Untitled Paper"}
          </h3>
        )}
      </div>

      {/* Body with metadata and abstract */}
      <div className="px-6 py-4 space-y-3">
        {/* Authors */}
        <div className="flex items-start gap-2">
          <span className="text-gray-500 font-medium min-w-[80px]">Authors:</span>
          <span className="text-gray-700 flex-1">{formatAuthors()}</span>
        </div>

        {/* Venue */}
        <div className="flex items-start gap-2">
          <span className="text-gray-500 font-medium min-w-[80px]">Venue:</span>
          <span className="text-gray-700 flex-1">{venueName}</span>
        </div>

        {/* Publication Date */}
        <div className="flex items-start gap-2">
          <span className="text-gray-500 font-medium min-w-[80px]">Published:</span>
          <span className="text-gray-700">{formatDate()}</span>
        </div>

        {/* Citation Count */}
        {props.citationCount !== undefined && (
          <div className="flex items-center gap-2">
            <span className="text-gray-500 font-medium min-w-[80px]">Citations:</span>
            <div className="flex items-center gap-1.5">
              <svg 
                className="w-4 h-4 text-amber-500" 
                fill="currentColor" 
                viewBox="0 0 20 20"
              >
                <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
              </svg>
              <span className="text-gray-900 font-semibold">{props.citationCount.toLocaleString()}</span>
            </div>
          </div>
        )}

        {/* Abstract */}
        <div className="pt-3 border-t border-gray-100">
          <p className="text-gray-500 font-medium mb-2">Abstract:</p>
          <p className="text-gray-600 text-sm leading-relaxed">
            {truncateAbstract(props.abstract)}
          </p>
        </div>
      </div>

      {/* Footer with link */}
      {props.url && (
        <div className="px-6 py-3 bg-gray-50 border-t border-gray-100">
          <a 
            href={props.url} 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-sm text-indigo-600 hover:text-indigo-700 font-medium inline-flex items-center gap-1"
          >
            View on Semantic Scholar
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
        </div>
      )}
    </div>
  );
};
