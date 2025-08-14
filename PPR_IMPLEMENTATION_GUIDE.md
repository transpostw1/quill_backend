# PPR (Partial Prerendering) Implementation Guide

## 🎯 What is PPR?

**Partial Prerendering (PPR)** is a Next.js 15 feature that combines:

- **Static prerendering** for fast initial loads
- **Streaming** for dynamic content
- **Progressive enhancement** for better UX

## 🚀 Your RAG API Now Supports PPR!

### New Streaming Endpoints

Your RAG API now has two new streaming endpoints:

#### 1. `/query/ppr` - PPR-Optimized Streaming

```bash
curl -X POST http://192.168.0.120:8000/query/ppr \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers do I have?", "stream": true}'
```

**Features:**

- ✅ Immediate page shell response
- ✅ Progressive result streaming
- ✅ Status updates in real-time
- ✅ SQL generation display
- ✅ Partial results loading

#### 2. `/query/stream` - Full Streaming

```bash
curl -X POST http://192.168.0.120:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me top customers", "stream": true}'
```

**Features:**

- ✅ Detailed status updates
- ✅ Intent classification
- ✅ SQL generation
- ✅ Chunked result delivery
- ✅ Error handling

## 📊 Streaming Response Format

### PPR Endpoint Response Types:

```json
{
  "type": "shell",
  "data": {
    "title": "SISL Database Query",
    "ready": true
  }
}
```

```json
{
  "type": "status",
  "data": {
    "message": "Processing your question..."
  }
}
```

```json
{
  "type": "sql",
  "data": {
    "sql": "SELECT COUNT(*) FROM [SISL Live].[dbo].[ssil_UAT$Customer]"
  }
}
```

```json
{
  "type": "results",
  "data": {
    "rows": [...],
    "partial": true,
    "total_found": 72
  }
}
```

```json
{
  "type": "complete",
  "data": {
    "success": true,
    "row_count": 72,
    "columns": ["Customer Count"]
  }
}
```

## 🛠️ Implementation Steps

### 1. Deploy Updated API

```bash
# Run the deployment script
./deploy_ppr_api.sh

# Or on Windows
.\deploy_ppr_api.ps1
```

### 2. Test the Endpoints

```bash
# Test PPR endpoint
curl -X POST http://192.168.0.120:8000/query/ppr \
  -H "Content-Type: application/json" \
  -d '{"question": "hi"}' \
  --max-time 30

# Test streaming endpoint
curl -X POST http://192.168.0.120:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "How many customers?"}' \
  --max-time 30
```

### 3. Use with Next.js

```jsx
// app/page.js
import { Suspense } from "react";
import { PPRRAGPage } from "./ppr_rag_component";

export default function Page() {
  return (
    <Suspense fallback={<LoadingFallback />}>
      <PPRRAGPage />
    </Suspense>
  );
}
```

## 🎨 PPR Benefits for Your RAG System

### ✅ **Faster Initial Load**

- Page shell loads immediately
- Users see content while API processes
- No blank loading screens

### ✅ **Better User Experience**

- Progressive content reveal
- Real-time status updates
- Interactive while processing

### ✅ **SEO Friendly**

- Static content is indexed
- Meta tags are prerendered
- Better Core Web Vitals

### ✅ **Scalable Performance**

- Reduces perceived loading time
- Handles large result sets gracefully
- Optimized for mobile

## 🔧 Configuration

### Next.js Configuration

```js
// next.config.js
export const config = {
  experimental: {
    ppr: true, // Enable PPR
  },
};
```

### Environment Variables

```env
# .env.local
NEXT_PUBLIC_RAG_API_URL=http://192.168.0.120:8000
NEXT_PUBLIC_ENABLE_PPR=true
```

## 📱 Frontend Integration

### React Hook for PPR

```jsx
// hooks/usePPRQuery.js
import { useState, useEffect } from "react";

export function usePPRQuery() {
  const [streamData, setStreamData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const queryWithPPR = async (question) => {
    setIsLoading(true);

    const response = await fetch("/api/query/ppr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, stream: true }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = JSON.parse(line.slice(6));
          setStreamData(data);
        }
      }
    }

    setIsLoading(false);
  };

  return { queryWithPPR, streamData, isLoading };
}
```

### API Route Handler

```js
// app/api/query/ppr/route.js
export async function POST(request) {
  const { question } = await request.json();

  const response = await fetch("http://192.168.0.120:8000/query/ppr", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, stream: true }),
  });

  return new Response(response.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
```

## 🧪 Testing PPR

### Manual Testing

```bash
# Test with curl
curl -N -X POST http://192.168.0.120:8000/query/ppr \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me top 5 customers"}' \
  --max-time 30
```

### Automated Testing

```js
// tests/ppr.test.js
import { test, expect } from "@playwright/test";

test("PPR streaming works correctly", async ({ page }) => {
  await page.goto("/");

  // Type a query
  await page.fill("textarea", "How many customers do I have?");
  await page.click('button[type="submit"]');

  // Check for immediate shell response
  await expect(page.locator("h1")).toBeVisible();

  // Check for streaming results
  await expect(page.locator(".bg-green-50")).toBeVisible();
});
```

## 📈 Performance Monitoring

### Vercel Analytics

```js
// Track PPR performance
export function reportPPRPerformance(startTime, endTime) {
  const duration = endTime - startTime;

  // Send to analytics
  fetch("/api/analytics", {
    method: "POST",
    body: JSON.stringify({
      type: "ppr_performance",
      duration,
      timestamp: Date.now(),
    }),
  });
}
```

### API Monitoring

```bash
# Monitor streaming endpoints
curl -w "@curl-format.txt" -X POST http://192.168.0.120:8000/query/ppr \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}' \
  -o /dev/null
```

## 🚨 Troubleshooting

### Common Issues

1. **Streaming not working**

   - Check if Ollama is running
   - Verify service status: `sudo systemctl status sisl-rag-api`
   - Check logs: `sudo journalctl -u sisl-rag-api -f`

2. **PPR not enabled**

   - Ensure Next.js 15+ is installed
   - Check `next.config.js` for PPR flag
   - Verify experimental features are enabled

3. **CORS issues**
   - Check API CORS configuration
   - Verify frontend URL is allowed
   - Test with different origins

### Debug Commands

```bash
# Check API health
curl http://192.168.0.120:8000/health

# Test streaming endpoint
curl -N http://192.168.0.120:8000/query/ppr \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'

# Monitor service logs
sudo journalctl -u sisl-rag-api -f
```

## 🎉 Summary

Your RAG API now supports **true PPR** with:

- ✅ **Streaming responses** for real-time updates
- ✅ **Progressive loading** for better UX
- ✅ **Intent classification** for smart responses
- ✅ **SQL generation** with live display
- ✅ **Error handling** with graceful fallbacks
- ✅ **Performance optimization** for large datasets

**The combination gives you the best of both worlds: fast initial loads and powerful RAG functionality!** 🚀
