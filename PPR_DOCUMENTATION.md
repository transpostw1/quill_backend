# Partial Prerendering (PPR) with SISL RAG API

## What is PPR?

Partial Prerendering (PPR) is a Next.js 15 feature that allows you to:

- **Prerender** static parts of your page at build time
- **Stream** dynamic parts as they become available
- **Improve performance** by showing content faster

## How PPR Works with Your RAG API

### 1. Static Parts (Prerendered)

These parts are rendered at build time and served immediately:

```jsx
// This gets prerendered
export default function Page() {
  return (
    <div>
      <h1>SISL Database Query</h1>
      <p>Ask questions about your data</p>
      <ExampleQuestions /> {/* Static content */}
      <DatabaseQueryForm /> {/* Dynamic - streams */}
    </div>
  );
}
```

### 2. Dynamic Parts (Streamed)

These parts stream in as data becomes available:

```jsx
// This streams with PPR
function DatabaseQueryForm() {
  const [query, setQuery] = useState("");

  const handleSubmit = async () => {
    // This calls your RAG API
    const response = await fetch("/api/query", {
      method: "POST",
      body: JSON.stringify({ question: query }),
    });
    // Results stream in as they're processed
  };
}
```

## Benefits for Your RAG Application

### ✅ **Faster Initial Load**

- Page shell loads immediately (title, form, examples)
- Users see content while RAG API processes queries

### ✅ **Better User Experience**

- No blank loading screens
- Progressive content reveal
- Interactive while processing

### ✅ **SEO Friendly**

- Static content is indexed
- Meta tags and structure are prerendered

## Implementation Steps

### 1. Enable PPR in Next.js

```js
// next.config.js
export const config = {
  experimental: {
    ppr: true,
  },
};
```

### 2. Structure Your Components

```jsx
// Static (prerendered)
function StaticHeader() {
  return <h1>Database Query Tool</h1>;
}

// Dynamic (streams)
function QueryForm() {
  // Client component that calls your API
}
```

### 3. Use Suspense Boundaries

```jsx
import { Suspense } from "react";

export default function Page() {
  return (
    <div>
      <StaticHeader /> {/* Prerendered */}
      <Suspense fallback={<Loading />}>
        <QueryForm /> {/* Streams */}
      </Suspense>
    </div>
  );
}
```

## API Integration

Your RAG API works perfectly with PPR:

### Request Flow:

1. **Static content** loads immediately
2. User types question in form
3. **Dynamic request** sent to your RAG API
4. **Results stream** back to UI

### Response Handling:

```jsx
// Your API response structure works with PPR
const result = await fetch('/api/query', {
  method: 'POST',
  body: JSON.stringify({ question: "How many customers?" })
});

// Response format:
{
  "question": "How many customers?",
  "generated_sql": "SELECT COUNT(*) FROM [SISL Live].[dbo].[SISL$Customer]",
  "results": [{"": "74"}],
  "columns": [""],
  "row_count": 1,
  "success": true
}
```

## Performance Benefits

### Before PPR:

- User sees blank page
- Waits for full page load
- Then can interact

### After PPR:

- User sees page immediately
- Can start typing while page loads
- Results appear progressively

## Deployment Considerations

### Vercel Deployment:

```bash
# Your RAG API (separate server)
python api.py  # Runs on port 8000

# Next.js app (Vercel)
vercel deploy  # Connects to your API
```

### Environment Variables:

```env
# .env.local
NEXT_PUBLIC_API_URL=http://your-rag-api-url:8000
```

## Example Usage Patterns

### 1. Progressive Enhancement

```jsx
// Static shell loads first
<PageHeader />
<Navigation />

// Dynamic content streams
<Suspense fallback={<QueryFormSkeleton />}>
  <DatabaseQueryForm />
</Suspense>
```

### 2. Streaming Results

```jsx
// Results appear as they're processed
function QueryResults({ promise }) {
  const result = use(promise); // React's use() hook
  return <DataTable data={result.results} />;
}
```

### 3. Error Boundaries

```jsx
// Graceful error handling
<ErrorBoundary fallback={<ErrorComponent />}>
  <Suspense fallback={<Loading />}>
    <QueryForm />
  </Suspense>
</ErrorBoundary>
```

## Best Practices

### ✅ Do:

- Keep static content in server components
- Use Suspense for dynamic content
- Handle loading and error states
- Optimize API response times

### ❌ Don't:

- Put everything in client components
- Block rendering with synchronous calls
- Forget error boundaries
- Ignore loading states

## Testing PPR

### Development:

```bash
npm run dev
# Check Network tab for streaming
```

### Production:

```bash
npm run build
npm run start
# Verify static parts are prerendered
```

## Monitoring

### Vercel Analytics:

- Track Core Web Vitals
- Monitor streaming performance
- Measure user engagement

### API Monitoring:

- Track RAG API response times
- Monitor Qdrant and Ollama performance
- Log query success rates

## Summary

Your RAG API is **perfectly compatible** with PPR because:

1. **Static parts** (UI, forms) prerender instantly
2. **Dynamic parts** (API calls) stream as needed
3. **User experience** improves dramatically
4. **Performance** benefits from progressive loading

The combination gives you the best of both worlds: fast initial loads and powerful RAG functionality!
