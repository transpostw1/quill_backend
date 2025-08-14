// Next.js PPR Component for SISL RAG API
'use client';

import { useState, useEffect, use } from 'react';
import { Suspense } from 'react';

// Static Shell Component (Prerendered)
function StaticShell() {
  return (
    <div className="bg-white shadow-lg rounded-lg p-6 mb-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-4">
        SISL Database Query Tool
      </h1>
      <p className="text-gray-600 mb-4">
        Ask questions about your data in natural language. The system will convert your questions to SQL and return results.
      </p>
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <h3 className="font-semibold text-blue-800 mb-2">Example Questions:</h3>
        <ul className="text-blue-700 space-y-1">
          <li>• "How many customers do I have?"</li>
          <li>• "Show me top customers by sales"</li>
          <li>• "What are my total sales this month?"</li>
          <li>• "List all products over $100"</li>
        </ul>
      </div>
    </div>
  );
}

// Loading Component
function LoadingFallback() {
  return (
    <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
      <div className="animate-pulse space-y-4">
        <div className="h-4 bg-gray-200 rounded w-3/4"></div>
        <div className="h-4 bg-gray-200 rounded w-1/2"></div>
        <div className="h-4 bg-gray-200 rounded w-5/6"></div>
      </div>
    </div>
  );
}

// Streaming Query Form Component
function StreamingQueryForm() {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamData, setStreamData] = useState(null);
  const [messages, setMessages] = useState([]);

  const handleStreamingQuery = async (question) => {
    setIsLoading(true);
    setMessages([]);
    
    try {
      const response = await fetch('http://192.168.0.120:8000/query/ppr', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question, stream: true })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              setMessages(prev => [...prev, data]);
              
              // Handle different message types
              switch (data.type) {
                case 'shell':
                  console.log('Page shell ready:', data.data);
                  break;
                case 'status':
                  console.log('Status:', data.data.message);
                  break;
                case 'sql':
                  console.log('Generated SQL:', data.data.sql);
                  break;
                case 'results':
                  if (data.data.rows) {
                    setStreamData(prev => ({
                      ...prev,
                      results: [...(prev?.results || []), ...data.data.rows],
                      partial: data.data.partial
                    }));
                  }
                  break;
                case 'conversation':
                  setStreamData({ response: data.data.response, isConversation: true });
                  break;
                case 'complete':
                  console.log('Query completed:', data.data);
                  break;
                case 'error':
                  console.error('Error:', data.data.error);
                  break;
              }
            } catch (e) {
              console.error('Error parsing stream data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      setMessages(prev => [...prev, { type: 'error', data: { error: error.message } }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    handleStreamingQuery(query);
  };

  return (
    <div className="space-y-6">
      {/* Query Form */}
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
          {isLoading ? 'Processing...' : 'Ask Question'}
        </button>
      </form>

      {/* Status Messages */}
      {messages.length > 0 && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <h3 className="font-semibold text-gray-800 mb-2">Processing Status:</h3>
          <div className="space-y-2">
            {messages
              .filter(msg => msg.type === 'status')
              .map((msg, index) => (
                <div key={index} className="text-sm text-gray-600">
                  {msg.data.message}
                </div>
              ))}
          </div>
        </div>
      )}

      {/* SQL Display */}
      {messages.find(msg => msg.type === 'sql') && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h3 className="font-semibold text-yellow-800 mb-2">Generated SQL:</h3>
          <pre className="text-sm text-yellow-700 bg-yellow-100 p-3 rounded overflow-x-auto">
            {messages.find(msg => msg.type === 'sql')?.data.sql}
          </pre>
        </div>
      )}

      {/* Results Display */}
      {streamData && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h3 className="font-semibold text-green-800 mb-2">
            {streamData.isConversation ? 'Response:' : 'Results:'}
          </h3>
          
          {streamData.isConversation ? (
            <p className="text-green-700">{streamData.response}</p>
          ) : (
            <div>
              <p className="text-green-700 mb-3">
                Found {streamData.results?.length || 0} results
                {streamData.partial && ' (loading more...)'}
              </p>
              
              {streamData.results && streamData.results.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-green-200">
                    <thead className="bg-green-100">
                      <tr>
                        {Object.keys(streamData.results[0]).map((key) => (
                          <th key={key} className="px-6 py-3 text-left text-xs font-medium text-green-800 uppercase tracking-wider">
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-green-200">
                      {streamData.results.map((row, index) => (
                        <tr key={index}>
                          {Object.values(row).map((value, cellIndex) => (
                            <td key={cellIndex} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                              {value}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Error Display */}
      {messages.find(msg => msg.type === 'error') && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="font-semibold text-red-800 mb-2">Error:</h3>
          <p className="text-red-700">
            {messages.find(msg => msg.type === 'error')?.data.error}
          </p>
        </div>
      )}
    </div>
  );
}

// Main PPR Page Component
export default function PPRRAGPage() {
  return (
    <div className="container mx-auto p-6 max-w-4xl">
      {/* Static Shell (Prerendered) */}
      <StaticShell />
      
      {/* Dynamic Content (Streams with PPR) */}
      <Suspense fallback={<LoadingFallback />}>
        <StreamingQueryForm />
      </Suspense>
    </div>
  );
}

// Alternative: Server Component with PPR
export function PPRServerComponent() {
  return (
    <div className="container mx-auto p-6 max-w-4xl">
      {/* This gets prerendered */}
      <StaticShell />
      
      {/* This streams */}
      <Suspense fallback={<LoadingFallback />}>
        <StreamingQueryForm />
      </Suspense>
    </div>
  );
} 