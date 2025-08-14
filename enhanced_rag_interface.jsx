// Enhanced RAG Interface - DB-GPT Inspired
'use client';

import { useState, useEffect, use } from 'react';
import { Suspense } from 'react';

// Static Shell Component (Prerendered)
function StaticShell() {
  return (
    <div className="bg-white shadow-lg rounded-lg p-6 mb-6">
      <h1 className="text-3xl font-bold text-gray-900 mb-4">
        Enhanced SISL RAG System
      </h1>
      <p className="text-gray-600 mb-4">
        DB-GPT inspired system for querying both database and documents using natural language.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-800 mb-2">Database Queries:</h3>
          <ul className="text-blue-700 space-y-1 text-sm">
            <li>• "How many customers do I have?"</li>
            <li>• "Show me top customers by sales"</li>
            <li>• "What are my total sales this month?"</li>
          </ul>
        </div>
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h3 className="font-semibold text-green-800 mb-2">Document Queries:</h3>
          <ul className="text-green-700 space-y-1 text-sm">
            <li>• "What does the contract say about payment?"</li>
            <li>• "Show me the invoice details"</li>
            <li>• "Find documents about customer agreements"</li>
          </ul>
        </div>
      </div>
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
        <h3 className="font-semibold text-purple-800 mb-2">Hybrid Queries:</h3>
        <ul className="text-purple-700 space-y-1 text-sm">
          <li>• "Show me sales data and related contracts"</li>
          <li>• "Customer information with their agreements"</li>
          <li>• "Financial reports and supporting documents"</li>
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

// Document Upload Component
function DocumentUpload() {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setUploadResult(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://192.168.0.120:8000/upload-document', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      setUploadResult(result);
      
      if (result.status === 'success') {
        setFile(null);
        // Reset file input
        const fileInput = document.getElementById('file-input');
        if (fileInput) fileInput.value = '';
      }
    } catch (error) {
      setUploadResult({
        status: 'error',
        message: `Upload failed: ${error.message}`
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="bg-white shadow-lg rounded-lg p-6 mb-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload Documents</h2>
      
      <div className="space-y-4">
        <div>
          <label htmlFor="file-input" className="block text-sm font-medium text-gray-700 mb-2">
            Select Document:
          </label>
          <input
            id="file-input"
            type="file"
            onChange={handleFileChange}
            accept=".pdf,.doc,.docx,.txt,.xlsx,.xls,.pptx,.ppt,.jpg,.jpeg,.png"
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
        </div>
        
        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {uploading ? 'Uploading...' : 'Upload Document'}
        </button>
        
        {uploadResult && (
          <div className={`p-4 rounded-lg ${
            uploadResult.status === 'success' 
              ? 'bg-green-50 border border-green-200 text-green-700' 
              : 'bg-red-50 border border-red-200 text-red-700'
          }`}>
            <p className="font-semibold">{uploadResult.status === 'success' ? '✅ Success' : '❌ Error'}</p>
            <p className="text-sm">{uploadResult.message}</p>
          </div>
        )}
      </div>
      
      <div className="mt-4 text-sm text-gray-600">
        <p className="font-semibold mb-2">Supported Formats:</p>
        <ul className="space-y-1">
          <li>• PDF, Word, Excel, PowerPoint</li>
          <li>• Text files (.txt)</li>
          <li>• Images (JPEG, PNG) - uses OCR</li>
        </ul>
      </div>
    </div>
  );
}

// Enhanced Query Form Component
function EnhancedQueryForm() {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamData, setStreamData] = useState(null);
  const [messages, setMessages] = useState([]);
  const [includeDocuments, setIncludeDocuments] = useState(true);
  const [includeDatabase, setIncludeDatabase] = useState(true);

  const handleStreamingQuery = async (question) => {
    setIsLoading(true);
    setMessages([]);
    setStreamData(null);
    
    try {
      const response = await fetch('http://192.168.0.120:8000/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          question, 
          stream: true,
          include_documents: includeDocuments,
          include_database: includeDatabase
        })
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
                      partial: data.data.partial,
                      source: data.data.source
                    }));
                  }
                  break;
                case 'documents':
                  if (data.data.documents) {
                    setStreamData(prev => ({
                      ...prev,
                      documents: data.data.documents,
                      documentSource: data.data.source
                    }));
                  }
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
            Ask a question about your data or documents:
          </label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., How many customers do I have? What does the contract say about payment terms?"
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            rows={3}
            disabled={isLoading}
          />
        </div>
        
        {/* Query Options */}
        <div className="flex flex-wrap gap-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={includeDatabase}
              onChange={(e) => setIncludeDatabase(e.target.checked)}
              className="mr-2"
            />
            <span className="text-sm text-gray-700">Include Database</span>
          </label>
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={includeDocuments}
              onChange={(e) => setIncludeDocuments(e.target.checked)}
              className="mr-2"
            />
            <span className="text-sm text-gray-700">Include Documents</span>
          </label>
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

      {/* Database Results Display */}
      {streamData?.results && streamData.results.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="font-semibold text-blue-800 mb-2">
            Database Results ({streamData.source}):
          </h3>
          
          <p className="text-blue-700 mb-3">
            Found {streamData.results.length} results
            {streamData.partial && ' (loading more...)'}
          </p>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-blue-200">
              <thead className="bg-blue-100">
                <tr>
                  {Object.keys(streamData.results[0]).map((key) => (
                    <th key={key} className="px-6 py-3 text-left text-xs font-medium text-blue-800 uppercase tracking-wider">
                      {key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-blue-200">
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
        </div>
      )}

      {/* Document Results Display */}
      {streamData?.documents && streamData.documents.length > 0 && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h3 className="font-semibold text-green-800 mb-2">
            Document Results ({streamData.documentSource}):
          </h3>
          
          <div className="space-y-4">
            {streamData.documents.map((doc, index) => (
              <div key={index} className="bg-white p-4 rounded-lg border border-green-200">
                <div className="flex justify-between items-start mb-2">
                  <span className="text-sm font-medium text-green-700">
                    Source: {doc.source}
                  </span>
                  <span className="text-xs text-gray-500">
                    Chunk {doc.chunk_id}
                  </span>
                </div>
                <p className="text-gray-800 text-sm leading-relaxed">
                  {doc.content}
                </p>
              </div>
            ))}
          </div>
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

// Main Enhanced RAG Page Component
export default function EnhancedRAGPage() {
  return (
    <div className="container mx-auto p-6 max-w-6xl">
      {/* Static Shell (Prerendered) */}
      <StaticShell />
      
      {/* Document Upload */}
      <DocumentUpload />
      
      {/* Dynamic Content (Streams with PPR) */}
      <Suspense fallback={<LoadingFallback />}>
        <EnhancedQueryForm />
      </Suspense>
    </div>
  );
}

// Alternative: Server Component with PPR
export function EnhancedRAGServerComponent() {
  return (
    <div className="container mx-auto p-6 max-w-6xl">
      {/* This gets prerendered */}
      <StaticShell />
      <DocumentUpload />
      
      {/* This streams */}
      <Suspense fallback={<LoadingFallback />}>
        <EnhancedQueryForm />
      </Suspense>
    </div>
  );
} 