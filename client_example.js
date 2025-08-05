// Example client for Next.js application to consume the SISL RAG API

const API_BASE_URL = 'http://localhost:8000'; // Change this to your API URL

/**
 * Query the database using natural language
 * @param {string} question - Natural language question
 * @param {number} maxResults - Maximum number of schema results to consider (default: 5)
 * @returns {Promise<Object>} - Query response
 */
async function queryDatabase(question, maxResults = 5) {
    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                max_results: maxResults
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error querying database:', error);
        throw error;
    }
}

/**
 * Get health status of the API
 * @returns {Promise<Object>} - Health status
 */
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error checking health:', error);
        throw error;
    }
}

/**
 * Get list of available tables
 * @returns {Promise<Object>} - Tables list
 */
async function getTables() {
    try {
        const response = await fetch(`${API_BASE_URL}/tables`);
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error getting tables:', error);
        throw error;
    }
}

// Example usage in Next.js
export async function handleUserQuery(userQuestion) {
    try {
        // Check API health first
        const health = await checkHealth();
        console.log('API Health:', health);

        // Query the database
        const result = await queryDatabase(userQuestion);
        
        if (result.success) {
            return {
                success: true,
                question: result.question,
                sql: result.generated_sql,
                results: result.results,
                columns: result.columns,
                rowCount: result.row_count,
                message: `Found ${result.row_count} results for your query.`
            };
        } else {
            return {
                success: false,
                error: result.error,
                message: 'Failed to process your query.'
            };
        }
    } catch (error) {
        return {
            success: false,
            error: error.message,
            message: 'Failed to connect to the API.'
        };
    }
}

// Example React component usage
export function DatabaseQueryComponent() {
    const [query, setQuery] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        
        try {
            const response = await handleUserQuery(query);
            setResult(response);
        } catch (error) {
            setResult({
                success: false,
                error: error.message,
                message: 'An error occurred.'
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask a question about your data..."
                    disabled={loading}
                />
                <button type="submit" disabled={loading}>
                    {loading ? 'Querying...' : 'Ask Question'}
                </button>
            </form>

            {result && (
                <div>
                    {result.success ? (
                        <div>
                            <h3>Results:</h3>
                            <p><strong>SQL:</strong> {result.sql}</p>
                            <p><strong>Message:</strong> {result.message}</p>
                            {result.results.length > 0 && (
                                <table>
                                    <thead>
                                        <tr>
                                            {result.columns.map((col, index) => (
                                                <th key={index}>{col}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {result.results.map((row, rowIndex) => (
                                            <tr key={rowIndex}>
                                                {result.columns.map((col, colIndex) => (
                                                    <td key={colIndex}>{row[col]}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            )}
                        </div>
                    ) : (
                        <div style={{color: 'red'}}>
                            <h3>Error:</h3>
                            <p>{result.message}</p>
                            {result.error && <p>Details: {result.error}</p>}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

// Example API routes for Next.js (pages/api/query.js)
export async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    try {
        const { question, max_results = 5 } = req.body;

        if (!question) {
            return res.status(400).json({ error: 'Question is required' });
        }

        const result = await queryDatabase(question, max_results);
        res.status(200).json(result);
    } catch (error) {
        res.status(500).json({ 
            error: 'Internal server error',
            message: error.message 
        });
    }
} 