import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { PaperListComponent, PaperComponent } from './ui';

// Mock the toast from sonner
vi.mock('sonner', () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

describe('PaperListComponent', () => {
  const mockPapers = [
    {
      paperId: 'paper-1',
      title: 'Test Paper 1',
      authors: [{ name: 'Author One' }, { name: 'Author Two' }],
      venue: 'Test Conference',
      year: 2024,
      abstract: 'This is a test abstract for paper 1',
      citationCount: 100,
      url: 'https://example.com/paper1',
    },
    {
      paperId: 'paper-2',
      title: 'Test Paper 2',
      authors: ['Author Three', 'Author Four'],
      publicationVenue: { name: 'Test Journal', type: 'journal' },
      publicationDate: '2023-01-15',
      year: 2023,
      abstract: 'This is a test abstract for paper 2',
      citationCount: 50,
      url: 'https://example.com/paper2',
    },
  ];

  beforeEach(() => {
    // Mock fetch using Vitest's stubGlobal
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  describe('Rendering', () => {
    it('should render the component with papers', () => {
      render(<PaperListComponent papers={mockPapers} />);

      expect(screen.getByText('📚 Found 2 Papers')).toBeInTheDocument();
      expect(screen.getByText('Test Paper 1')).toBeInTheDocument();
      expect(screen.getByText('Test Paper 2')).toBeInTheDocument();
    });

    it('should render "No papers found" when papers array is empty', () => {
      render(<PaperListComponent papers={[]} />);

      expect(screen.getByText('No papers found.')).toBeInTheDocument();
    });

    it('should render "No papers found" when papers is undefined', () => {
      render(<PaperListComponent papers={undefined as any} />);

      expect(screen.getByText('No papers found.')).toBeInTheDocument();
    });

    it('should display correct paper count in singular form', () => {
      render(<PaperListComponent papers={[mockPapers[0]]} />);

      expect(screen.getByText('📚 Found 1 Paper')).toBeInTheDocument();
    });

    it('should show selection hint text when no papers are selected', () => {
      render(<PaperListComponent papers={mockPapers} />);

      expect(screen.getByText('Select papers to save')).toBeInTheDocument();
    });
  });

  describe('Paper Selection', () => {
    it('should allow selecting a single paper', async () => {
      const user = userEvent.setup();
      render(<PaperListComponent papers={mockPapers} />);

      const checkboxes = screen.getAllByRole('checkbox');
      await user.click(checkboxes[0]);

      expect(screen.getByText('1 selected')).toBeInTheDocument();
    });

    it('should allow selecting multiple papers', async () => {
      const user = userEvent.setup();
      render(<PaperListComponent papers={mockPapers} />);

      const checkboxes = screen.getAllByRole('checkbox');
      await user.click(checkboxes[0]);
      await user.click(checkboxes[1]);

      expect(screen.getByText('2 selected')).toBeInTheDocument();
    });

    it('should allow deselecting papers', async () => {
      const user = userEvent.setup();
      render(<PaperListComponent papers={mockPapers} />);

      const checkboxes = screen.getAllByRole('checkbox');

      // Select both
      await user.click(checkboxes[0]);
      await user.click(checkboxes[1]);
      expect(screen.getByText('2 selected')).toBeInTheDocument();

      // Deselect one
      await user.click(checkboxes[0]);
      expect(screen.getByText('1 selected')).toBeInTheDocument();
    });

    it('should show ingest button only when papers are selected', async () => {
      const user = userEvent.setup();
      render(<PaperListComponent papers={mockPapers} />);

      // Initially, no ingest button
      expect(screen.queryByText(/Ingest/)).not.toBeInTheDocument();

      // Select a paper
      const checkboxes = screen.getAllByRole('checkbox');
      await user.click(checkboxes[0]);

      // Now ingest button should appear
      expect(screen.getByText('Ingest 1 Paper')).toBeInTheDocument();
    });
  });

  describe('Ingest Functionality', () => {
    it('should use the correct API URL from environment variable', async () => {
      const user = userEvent.setup();
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ tasks: [] }),
      });
      vi.stubGlobal('fetch', mockFetch);

      render(<PaperListComponent papers={mockPapers} />);

      const checkboxes = screen.getAllByRole('checkbox');
      await user.click(checkboxes[0]);

      const ingestButton = screen.getByText('Ingest 1 Paper');
      await user.click(ingestButton);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:2024/ingest',
          expect.any(Object)
        );
      });
    });

    it('should send correct payload when ingesting papers', async () => {
      const user = userEvent.setup();
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ tasks: [{ paperId: 'paper-1', taskId: 'task-1' }] }),
      });
      vi.stubGlobal('fetch', mockFetch);

      render(<PaperListComponent papers={mockPapers} />);

      const checkboxes = screen.getAllByRole('checkbox');
      await user.click(checkboxes[0]);

      const ingestButton = screen.getByText('Ingest 1 Paper');
      await user.click(ingestButton);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          'http://localhost:2024/ingest',
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              papers: [mockPapers[0]],
            }),
          }
        );
      });
    });

    it('should show success toast on successful ingestion', async () => {
      const user = userEvent.setup();
      const { toast } = await import('sonner');

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ tasks: [{ paperId: 'paper-1', taskId: 'task-1' }] }),
      });
      vi.stubGlobal('fetch', mockFetch);

      render(<PaperListComponent papers={mockPapers} />);

      const checkboxes = screen.getAllByRole('checkbox');
      await user.click(checkboxes[0]);

      const ingestButton = screen.getByText('Ingest 1 Paper');
      await user.click(ingestButton);

      await waitFor(() => {
        expect(toast.success).toHaveBeenCalledWith(
          'Papers Submitted for Ingestion',
          expect.objectContaining({
            description: expect.stringContaining('1 paper(s)'),
          })
        );
      });
    });

    it('should clear selection after successful ingestion', async () => {
      const user = userEvent.setup();
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ tasks: [] }),
      });
      vi.stubGlobal('fetch', mockFetch);

      render(<PaperListComponent papers={mockPapers} />);

      const checkboxes = screen.getAllByRole('checkbox');
      await user.click(checkboxes[0]);

      expect(screen.getByText('1 selected')).toBeInTheDocument();

      const ingestButton = screen.getByText('Ingest 1 Paper');
      await user.click(ingestButton);

      await waitFor(() => {
        expect(screen.getByText('Select papers to save')).toBeInTheDocument();
      });
    });

    it('should show error toast on failed ingestion', async () => {
      const user = userEvent.setup();
      const { toast } = await import('sonner');

      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        text: async () => 'Server error',
      });
      vi.stubGlobal('fetch', mockFetch);

      render(<PaperListComponent papers={mockPapers} />);

      const checkboxes = screen.getAllByRole('checkbox');
      await user.click(checkboxes[0]);

      const ingestButton = screen.getByText('Ingest 1 Paper');
      await user.click(ingestButton);

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith(
          'Ingestion Failed',
          expect.objectContaining({
            description: expect.stringContaining('500'),
          })
        );
      });
    });

    it('should disable button and show loading state during ingestion', async () => {
      const user = userEvent.setup();

      // Create a promise that we can control
      let resolveIngest: (value: any) => void;
      const ingestPromise = new Promise((resolve) => {
        resolveIngest = resolve;
      });

      const mockFetch = vi.fn().mockReturnValue(ingestPromise);
      vi.stubGlobal('fetch', mockFetch);

      render(<PaperListComponent papers={mockPapers} />);

      const checkboxes = screen.getAllByRole('checkbox');
      await user.click(checkboxes[0]);

      const ingestButton = screen.getByText('Ingest 1 Paper');
      await user.click(ingestButton);

      // Check loading state
      await waitFor(() => {
        expect(screen.getByText('Ingesting...')).toBeInTheDocument();
      });

      const loadingButton = screen.getByText('Ingesting...').closest('button');
      expect(loadingButton).toBeDisabled();

      // Resolve the promise
      resolveIngest!({
        ok: true,
        json: async () => ({ tasks: [] }),
      });

      await waitFor(() => {
        expect(screen.queryByText('Ingesting...')).not.toBeInTheDocument();
      });
    });

    it('should handle network errors gracefully', async () => {
      const user = userEvent.setup();
      const { toast } = await import('sonner');

      const mockFetch = vi.fn().mockRejectedValue(new Error('Network error'));
      vi.stubGlobal('fetch', mockFetch);

      render(<PaperListComponent papers={mockPapers} />);

      const checkboxes = screen.getAllByRole('checkbox');
      await user.click(checkboxes[0]);

      const ingestButton = screen.getByText('Ingest 1 Paper');
      await user.click(ingestButton);

      await waitFor(() => {
        expect(toast.error).toHaveBeenCalledWith(
          'Ingestion Failed',
          expect.objectContaining({
            description: 'Network error',
          })
        );
      });
    });
  });
});

describe('PaperComponent', () => {
  const mockPaper = {
    paperId: 'paper-1',
    title: 'Test Paper',
    authors: [{ name: 'Author One' }, { name: 'Author Two' }],
    venue: 'Test Conference',
    year: 2024,
    abstract: 'This is a test abstract',
    citationCount: 100,
    url: 'https://example.com/paper',
  };

  it('should render paper details correctly', () => {
    const mockOnSelectChange = vi.fn();

    render(
      <PaperComponent
        {...mockPaper}
        isSelected={false}
        onSelectChange={mockOnSelectChange}
      />
    );

    expect(screen.getByText('Test Paper')).toBeInTheDocument();
    expect(screen.getByText(/Author One, Author Two/)).toBeInTheDocument();
    expect(screen.getByText('Test Conference')).toBeInTheDocument();
    expect(screen.getByText('2024')).toBeInTheDocument();
    expect(screen.getByText(/This is a test abstract/)).toBeInTheDocument();
  });

  it('should call onSelectChange when checkbox is clicked', async () => {
    const user = userEvent.setup();
    const mockOnSelectChange = vi.fn();

    render(
      <PaperComponent
        {...mockPaper}
        isSelected={false}
        onSelectChange={mockOnSelectChange}
      />
    );

    const checkbox = screen.getByRole('checkbox');
    await user.click(checkbox);

    expect(mockOnSelectChange).toHaveBeenCalledWith('paper-1', true);
  });

  it('should show selected badge when paper is selected', () => {
    const mockOnSelectChange = vi.fn();

    render(
      <PaperComponent
        {...mockPaper}
        isSelected={true}
        onSelectChange={mockOnSelectChange}
      />
    );

    expect(screen.getByText('Selected')).toBeInTheDocument();
  });

  it('should handle missing optional fields gracefully', () => {
    const minimalPaper = {
      paperId: 'paper-1',
    };
    const mockOnSelectChange = vi.fn();

    render(
      <PaperComponent
        {...minimalPaper}
        isSelected={false}
        onSelectChange={mockOnSelectChange}
      />
    );

    expect(screen.getByText('Untitled Paper')).toBeInTheDocument();
    expect(screen.getByText('Unknown authors')).toBeInTheDocument();
    expect(screen.getByText('Unknown venue')).toBeInTheDocument();
  });

  it('should truncate long abstracts', () => {
    const longAbstract = 'word '.repeat(250); // More than 200 words
    const mockOnSelectChange = vi.fn();

    render(
      <PaperComponent
        {...mockPaper}
        abstract={longAbstract}
        isSelected={false}
        onSelectChange={mockOnSelectChange}
      />
    );

    const abstractText = screen.getByText(/word/);
    expect(abstractText.textContent).toContain('...');
  });

  it('should render external link correctly', () => {
    const mockOnSelectChange = vi.fn();

    render(
      <PaperComponent
        {...mockPaper}
        isSelected={false}
        onSelectChange={mockOnSelectChange}
      />
    );

    const link = screen.getByRole('link', { name: 'Open paper' });
    expect(link).toHaveAttribute('href', 'https://example.com/paper');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  });

  it('should format citation count with locale string', () => {
    const mockOnSelectChange = vi.fn();

    render(
      <PaperComponent
        {...mockPaper}
        citationCount={1000}
        isSelected={false}
        onSelectChange={mockOnSelectChange}
      />
    );

    expect(screen.getByText('1,000')).toBeInTheDocument();
  });

  it('should handle string array authors', () => {
    const mockOnSelectChange = vi.fn();

    render(
      <PaperComponent
        {...mockPaper}
        authors={['Author One', 'Author Two', 'Author Three']}
        isSelected={false}
        onSelectChange={mockOnSelectChange}
      />
    );

    expect(screen.getByText(/Author One, Author Two, Author Three/)).toBeInTheDocument();
  });

  it('should prefer publicationVenue.name over venue', () => {
    const mockOnSelectChange = vi.fn();

    render(
      <PaperComponent
        {...mockPaper}
        venue="Old Venue"
        publicationVenue={{ name: 'New Venue', type: 'journal' }}
        isSelected={false}
        onSelectChange={mockOnSelectChange}
      />
    );

    expect(screen.getByText('New Venue')).toBeInTheDocument();
    expect(screen.queryByText('Old Venue')).not.toBeInTheDocument();
  });
});
