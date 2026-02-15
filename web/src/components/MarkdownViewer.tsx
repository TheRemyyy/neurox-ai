import { useEffect, useState } from 'react';
import { marked } from 'marked';
import hljs from 'highlight.js';
import katex from 'katex';
import 'highlight.js/styles/github-dark.css';
import 'katex/dist/katex.min.css';

interface MarkdownViewerProps {
  path: string;
}

function renderMathInMarkdown(md: string): { processed: string; blockPlaceholders: string[]; inlinePlaceholders: string[] } {
  const blockPlaceholders: string[] = [];
  const inlinePlaceholders: string[] = [];

  const replacer = (segment: string, inCodeBlock: boolean) => {
    if (inCodeBlock) return segment;
    let out = segment;
    out = out.replace(/\$\$([\s\S]*?)\$\$/g, (_, formula) => {
      const i = blockPlaceholders.length;
      blockPlaceholders.push(formula.trim());
      return `\u200B<!--MATH_BLOCK_${i}-->\u200B`;
    });
    out = out.replace(/\$([^\$\n]+)\$/g, (_, formula) => {
      const i = inlinePlaceholders.length;
      inlinePlaceholders.push(formula.trim());
      return `\u200B<!--MATH_INLINE_${i}-->\u200B`;
    });
    return out;
  };

  const parts = md.split(/(```[\s\S]*?```)/g);
  const processed = parts.map((part) => replacer(part, part.startsWith('```'))).join('');
  return { processed, blockPlaceholders, inlinePlaceholders };
}

function applyRenderedMath(
  html: string,
  blockPlaceholders: string[],
  inlinePlaceholders: string[]
): string {
  const opts = { throwOnError: false };
  let out = html;
  blockPlaceholders.forEach((formula, i) => {
    const rendered = katex.renderToString(formula, { ...opts, displayMode: true });
    const wrapped = `<span class="katex-block-wrapper">${rendered}</span>`;
    out = out.replace(`\u200B<!--MATH_BLOCK_${i}-->\u200B`, wrapped);
  });
  inlinePlaceholders.forEach((formula, i) => {
    const rendered = katex.renderToString(formula, { ...opts, displayMode: false });
    out = out.replace(`\u200B<!--MATH_INLINE_${i}-->\u200B`, rendered);
  });
  return out;
}

export const MarkdownViewer = ({ path }: MarkdownViewerProps) => {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(path)
      .then((res) => res.text())
      .then((text) => {
        const { processed: textWithPlaceholders, blockPlaceholders, inlinePlaceholders } = renderMathInMarkdown(text);
        const html = marked.parse(textWithPlaceholders, { breaks: true, gfm: true }) as string;
        const withMath = applyRenderedMath(html, blockPlaceholders, inlinePlaceholders);
        setContent(withMath);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load markdown:', err);
        setContent('<p className="text-red-500">Failed to load documentation segment.</p>');
        setLoading(false);
      });
  }, [path]);

  useEffect(() => {
    if (!loading) {
      document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block as HTMLElement);
      });
    }
  }, [content, loading]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-t-accent border-border rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div 
      className="prose prose-invert prose-pre:bg-surface prose-pre:border prose-pre:border-border max-w-none
                 prose-headings:font-semibold prose-headings:tracking-tight
                 prose-a:text-accent prose-a:no-underline hover:prose-a:underline
                 prose-code:text-accent prose-code:bg-surface prose-code:px-1 prose-code:rounded"
      dangerouslySetInnerHTML={{ __html: content }}
    />
  );
};
