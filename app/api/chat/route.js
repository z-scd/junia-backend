import { NextResponse } from "next/server";
import {
  GoogleGenerativeAIEmbeddings,
  ChatGoogleGenerativeAI,
} from "@langchain/google-genai";
import { createClient } from "@supabase/supabase-js";

import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";

export async function POST(request) {
  const { question } = await request.json();

  const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "gemini-embedding-001",
    apiKey: process.env.GEMINI_API_KEY,
  });

  const supabase = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SECRET_KEY,
  );

  const vectorStore = new SupabaseVectorStore(embeddings, {
    client: supabase,
    tableName: "documents",
    queryName: "match_documents",
  });

  const results = await vectorStore.similaritySearch(question, 4);

  const context = results.map((doc) => doc.pageContent).join("\n\n---\n\n");

  const prompt = `You are a helpful assistant answering questions using only the context below.
                  Answer in the same language the question was asked in (Bengali or English).
                  If the answer isn't in the context, say you don't know.
                  Context:
                  ${context}
                  Question: ${question}`;

  const model = new ChatGoogleGenerativeAI({
    model: "gemini-2.5-flash",
    apiKey: process.env.GEMINI_API_KEY,
  });

  const response = await model.invoke(prompt);

  return NextResponse.json({
    answer: response.content,
  });
}
