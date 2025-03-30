import 'dotenv/config'
import readline from 'readline'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { OpenAIEmbeddings } from '@langchain/openai'
import { CharacterTextSplitter } from 'langchain/text_splitter'
import { PDFLoader } from 'langchain/document_loaders/fs/pdf'
import { openai } from './openai.js'

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
})

const createStore = (docs) =>
  MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings())

const docsFromPDF = () => {
  const loader = new PDFLoader('./Junior Degree Works.pdf')
  return loader.loadAndSplit(
    new CharacterTextSplitter({
      separator: '. ',
      chunkSize: 2500,
      chunkOverlap: 200,
    })
  )
}

const loadStore = async () => {
  const pdfDocs = await docsFromPDF()
  return createStore(pdfDocs)
}

const initialQuery = async (store, messages) => {
  const initialQuestion = `Take a look at my degree works and tell me what classes I have to take to finish school. I want to take 15 credits per semester. Break it down and make sure there are no prereqs I missed. Create a schedule table. DO NOT make anything up. Check for pre requisite and make sure all classes are taken in order. Create me a table for every semester`

  const results = await store.similaritySearch(initialQuestion, 3)

  messages.push({ role: 'user', content: `Question: ${initialQuestion}\n\nContext: ${results.map(r => r.pageContent).join('\n')}` })

  const response = await openai.chat.completions.create({
    model: 'gpt-3.5-turbo-0125',
    temperature: 0,
    messages,
  })

  const reply = response.choices[0].message.content
  messages.push({ role: 'assistant', content: reply })

  console.log(`\n🤖 AI (Initial Response):\n${reply}\n`)
}

const chat = async () => {
  const store = await loadStore()
  const messages = [
    {
      role: 'system',
      content: 'You are a helpful academic advisor. Only use the provided context. NEVER make up anything.',
    },
  ]

  // Run the initial summary automatically
  await initialQuery(store, messages)

  // Start chat loop
  const promptUser = () => {
    rl.question('\n👤 You: ', async (userInput) => {
      if (userInput.toLowerCase() === 'exit') {
        console.log('👋 Exiting chat.')
        rl.close()
        return
      }

      const results = await store.similaritySearch(userInput, 3)

      messages.push({ role: 'user', content: `Question: ${userInput}\n\nContext: ${results.map(r => r.pageContent).join('\n')}` })

      const response = await openai.chat.completions.create({
        model: 'gpt-3.5-turbo-0125',
        temperature: 0,
        messages,
      })

      const reply = response.choices[0].message.content
      messages.push({ role: 'assistant', content: reply })

      console.log(`🤖 AI: ${reply}`)
      promptUser()
    })
  }

  promptUser()
}

chat()
