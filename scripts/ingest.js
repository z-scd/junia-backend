import "dotenv/config";
import { createClient } from "@supabase/supabase-js";

import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";

import fs from "fs/promises";

import path from "path";

import { fileURLToPath } from "url";

import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { BOOK_METADATA } from "../utils/metadata.js";

import { Document } from "langchain/document";

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SECRET_KEY,
);

const embeddings = new GoogleGenerativeAIEmbeddings({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-embedding-001",
});

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const dataDir = path.join(__dirname, "../data");

const files = await fs.readdir(dataDir);
const markdownFiles = files.filter((file) => file.endsWith(".md"));

const splitter = new RecursiveCharacterTextSplitter({
  chunkOverlap: 200,
  chunkSize: 1000,
  separators: ["\n\n", "\n", "। ", ". ", " ", ""],
});

for (const file of markdownFiles) {
  const content = await fs.readFile(path.join(dataDir, file), "utf8");

  const chunks = await splitter.splitText(content);

  const documents = chunks.map((chunk) => {
    return new Document({
      pageContent: chunk,
      metadata: {
        source: file,
        ...BOOK_METADATA[file],
      },
    });
  });

  await SupabaseVectorStore.fromDocuments(documents, embeddings, {
    client: supabase,
    tableName: "documents",
    queryName: "match_documents",
  });

  console.log(`Stored ${documents.length} chunks from ${file}`);
}
