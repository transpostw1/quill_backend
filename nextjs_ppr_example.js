// Next.js with Partial Prerendering (PPR) example for SISL RAG API

// 1. API Route (pages/api/query.js or app/api/query/route.js)
export async function POST(request) {
  try {
    const { question, max_results = 5 } = await request.json();
    
    if (!question) {
      return Response.json({ error: 'Question is required' }, { status: 400 });
    }

    // Call your RAG API
    const response = await fetch('http://localhost:8000/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        max_results
      })
    });

    const result = await response.json();
    return Response.json(result);
  } catch (error) {
    return Response.json({ 
      error: 'Internal server error',
      message: error.message 
    }, { status: 500 });
  }
}

// 2. Server Component (Static part - prerendered)
export default function DatabaseQueryPage() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">SISL Database Query</h1>
      <p className="text-gray-600 mb-4">
        Ask questions about your data in natural language
      </p>
      
      {/* Static content that gets prerendered */}
      <div className="bg-blue-50 p-4 rounded-lg mb-6">
        <h2 className="font-semibold mb-2">Example Questions:</h2>
        <ul className="list-disc list-inside space-y-1">
          <li>"How many customers do I have?"</li>
          <li>"Show me top customers by sales"</li>
          <li>"What are my recent orders?"</li>
          <li>"Count total sales this month"</li>
        </ul>
      </div>

      {/* Dynamic component that streams */}
      <DatabaseQueryForm />
    </div>
  );
}

// 3. Client Component (Dynamic part - streams with PPR)
'use client';

import { useState } from 'react';
import { use } from 'react';

// Suspense boundary for streaming
function QueryResults({ queryPromise }) {
  const result = use(queryPromise);
  
  if (!result.success) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <h3 className="font-semibold text-red-800">Error:</h3>
        <p className="text-red-700">{result.message}</p>
        {result.error && <p className="text-sm text-red-600">Details: {result.error}</p>}
      </div>
    );
  }

  return (
    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
      <h3 className="font-semibold text-green-800 mb-2">Results:</h3>
      <p className="text-sm text-gray-600 mb-2">
        <strong>SQL:</strong> {result.generated_sql}
      </p>
      <p className="text-green-700 mb-3">
        Found {result.row_count} results for your query.
      </p>
      
      {result.results.length > 0 && (
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white border border-gray-200">
            <thead>
              <tr className="bg-gray-50">
                {result.columns.map((col, index) => (
                  <th key={index} className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.results.map((row, rowIndex) => (
                <tr key={rowIndex} className="hover:bg-gray-50">
                  {result.columns.map((col, colIndex) => (
                    <td key={colIndex} className="px-4 py-2 text-sm text-gray-900 border-b">
                      {row[col]}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// Main form component with PPR streaming
export function DatabaseQueryForm() {
  const [query, setQuery] = useState('');
  const [queryPromise, setQueryPromise] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    
    // Create a promise that will be used by Suspense
    const promise = fetch('/api/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question: query })
    }).then(res => res.json());

    setQueryPromise(promise);
    setIsLoading(false);
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
            Ask a question about your data:
          </label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., How many customers do I have?"
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            rows={3}
            disabled={isLoading}
          />
        </div>
        <button
          type="submit"
          disabled={isLoading || !query.trim()}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Querying...' : 'Ask Question'}
        </button>
      </form>

      {/* Suspense boundary for streaming results */}
      {queryPromise && (
        <div className="mt-6">
          <Suspense fallback={
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span className="text-blue-700">Processing your query...</span>
              </div>
            </div>
          }>
            <QueryResults queryPromise={queryPromise} />
          </Suspense>
        </div>
      )}
    </div>
  );
}

// 4. Alternative: Using React Server Components with streaming
// app/page.js (App Router)
import { Suspense } from 'react';

// This component can be prerendered
export default function Page() {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">SISL Database Query</h1>
      
      {/* Static content */}
      <StaticContent />
      
      {/* Dynamic content that streams */}
      <Suspense fallback={<LoadingFallback />}>
        <DynamicQuerySection />
      </Suspense>
    </div>
  );
}

// Static component (prerendered)
function StaticContent() {
  return (
    <div className="bg-gray-50 p-4 rounded-lg mb-6">
      <h2 className="font-semibold mb-2">Available Tables:</h2>
      <p className="text-sm text-gray-600">
        Your database contains tables like SISL$Customer, SISL$Company, and many more.
        Ask questions about any of these tables using natural language.
      </p>
    </div>
  );
}

// Dynamic component (streams with PPR)
async function DynamicQuerySection() {
  // This could fetch initial data or show a form
  return <DatabaseQueryForm />;
}

function LoadingFallback() {
  return (
    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
      <div className="flex items-center space-x-2">
        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
        <span className="text-blue-700">Loading...</span>
      </div>
    </div>
  );
}

// 5. Configuration for PPR
// next.config.js
export const config = {
  experimental: {
    ppr: true, // Enable Partial Prerendering
  },
};

// 6. Loading and Error boundaries
// app/loading.js
export default function Loading() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
    </div>
  );
}

// app/error.js
'use client';

export default function Error({ error, reset }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen">
      <h2 className="text-xl font-semibold mb-4">Something went wrong!</h2>
      <p className="text-gray-600 mb-4">{error.message}</p>
      <button
        onClick={reset}
        className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
      >
        Try again
      </button>
    </div>
  );
} 