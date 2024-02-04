import "dotenv/config";

import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { formatDocumentsAsString } from "langchain/util/document";
import { CharacterTextSplitter } from "langchain/text_splitter";
import fs from "fs";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

const model = new ChatOpenAI({});

const text = fs.readFileSync("raw.txt", "utf-8");
const splitter = new CharacterTextSplitter({
  separator: ".",
  chunkOverlap: 3,
});
const documents = await splitter.createDocuments([text]);

const vectorStore = await HNSWLib.fromDocuments(
  documents,
  new OpenAIEmbeddings()
);

const retriever = vectorStore.asRetriever();

const prompt =
  PromptTemplate.fromTemplate(`Answer the question based only on the following context:
{context}

Question: {question}`);

const chain = RunnableSequence.from([
  {
    context: retriever.pipe(formatDocumentsAsString),
    question: new RunnablePassthrough(),
  },
  prompt,
  model,
  new StringOutputParser(),
]);

export const run = async () => {
  const stream = await chain.invoke("What do seals eat?");

  for await (const chunk of stream) {
    process.stdout.write(chunk);
  }
};

run();
