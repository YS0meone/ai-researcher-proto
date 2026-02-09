import '@testing-library/jest-dom';
import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';

// Cleanup after each test
afterEach(() => {
  cleanup();
});

// Mock environment variables for testing
// These provide predictable values regardless of your local .env file
// This ensures tests work in CI/CD and for other developers
vi.stubEnv('VITE_BACKEND_API_URL', 'http://localhost:2024');
vi.stubEnv('VITE_API_URL', 'http://localhost:2024');
vi.stubEnv('VITE_ASSISTANT_ID', 'agent');
