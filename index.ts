import "dotenv/config";

import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { formatDocumentsAsString } from "langchain/util/document";
import { CharacterTextSplitter } from "langchain/text_splitter";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import readline from "node:readline/promises";
import {
  RunnableSequence,
  RunnablePassthrough,
} from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const model = new ChatOpenAI({ verbose: false });

const splitter = new CharacterTextSplitter({
  separator: " ",
  chunkOverlap: 3,
});

const loader = new CheerioWebBaseLoader(
  "https://en.wikipedia.org/wiki/Pinniped"
);
const data = await loader.load();
const documents = await splitter.splitDocuments(data);

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
  const text = await rl.question(
    "AI: What do you want to know about pinnipeds? \n"
  );
  const stream = await chain.stream(text);
  console.log("");
  process.stdout.write("AI: ");
  for await (const chunk of stream) {
    process.stdout.write(chunk);
  }
  console.log("\n");
};

while (true) {
  await run();
}
