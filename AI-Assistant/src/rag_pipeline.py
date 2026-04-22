from typing import Dict, List
from pathlib import Path
from langchain_core.documents import Document
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager, VectorStoreNotFoundError
from src.retriever import Retriever
from src.generator import Generator
from src.youtube_processor import YouTubeProcessor
from src.recommender import RecommendationEngine
from src.presentation_maker import PresentationMaker
from config.settings import RAW_DATA_DIR
import logging
import os
import re

logger = logging.getLogger(__name__)

NOT_COVERED_PHRASES = [
    "not covered in the course materials",
    "غير مذكور في المواد الدراسية",
    "i can only answer questions based on the provided documents",
    "لا أستطيع الإجابة إلا بناءً على المستندات المقدمة",
]

def _is_not_covered(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in NOT_COVERED_PHRASES)

class RAGPipeline:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.retriever = Retriever(vector_store_manager=self.vector_store_manager)
        self.generator = Generator()
        self.youtube_processor = YouTubeProcessor()
        self.recommender = RecommendationEngine()
        self.presentation_maker = PresentationMaker()
        self.is_initialized = False

    def initialize(self, data_path: str = None):
        from config.settings import CHUNKS_CACHE_PATH
        if data_path is None:
            data_path = str(RAW_DATA_DIR)

        logger.info("Initializing RAG pipeline...")
        chunks = []
        if os.path.exists(CHUNKS_CACHE_PATH):
            logger.info(f"Loading chunks from cache: {CHUNKS_CACHE_PATH}")
            chunks = self.document_processor.load_chunks(CHUNKS_CACHE_PATH)

        if not chunks:
            logger.info("Processing documents (no cache found or cache empty)...")
            chunks = self.document_processor.process_courses_from_root(data_path)

            if not chunks:
                raise ValueError("No documents were processed. Check your data directory.")
            self.document_processor.save_chunks(chunks, CHUNKS_CACHE_PATH)

        logger.info("Creating/Updating vector store...")
        self.vector_store_manager.create_vector_store(chunks, overwrite=True)
        self.is_initialized = True
        logger.info("RAG pipeline initialized successfully!")

    def add_documents(self, source_path: str, course_name: str = None):
        """Add new documents to existing vector store."""
        logger.info(f"Adding documents from {source_path}")
        chunks = self.document_processor.process_documents(source_path, course_name=course_name)
        if not chunks:
            logger.warning("No new documents were processed")
            return
        self.vector_store_manager.add_documents(chunks)
        self.retriever.invalidate_bm25()
        logger.info("Documents added successfully!")

    # ================================================================== #
    # ★ الدالة الرئيسية المحدثة (Agentic Evaluate-and-Route + Smart Filter + Global Fallback) ★
    # ================================================================== #
    def query(
        self, 
        question: str, 
        history: list = None, 
        user_courses: List[str] = None,     
        selected_course: str = None,         
        forced_documents: List[Document] = None, 
        image_paths: List[str] = None
    ) -> Dict:
        
        if not self.is_initialized:
            logger.info("Pipeline not initialized, loading existing vector store...")
            try:
                self.vector_store_manager.load_vector_store()
                self.is_initialized = True
            except VectorStoreNotFoundError as e:
                logger.error(f"Failed to load vector store: {e}")
                return {"answer": "Vector store not found. Please run initialization first.", "sources": []}

        logger.info(f"Processing query: {question}")
        has_arabic = any('\u0600' <= char <= '\u06FF' for char in question)

        # -----------------------------------------------------------------
        # 0. Intent Detection & Query Cleaning
        # -----------------------------------------------------------------
        question_lower = question.lower()
        presentation_keywords = ["presentation", "slides", "powerpoint", "pptx", "make a presentation", "عرض تقديمي", "شرائح", "بوربوينت", "اعمل عرض", "سوي بريزنتيشن"]
        recommendation_keywords = ["recommend", "suggest", "more resources", "other courses", "another video", "مقترح", "ترشيح", "مصادر أخرى", "كورس آخر", "نرشح", "زيدني"]

        is_presentation = any(keyword in question_lower for keyword in presentation_keywords)
        is_recommendation = any(keyword in question_lower for keyword in recommendation_keywords)

        url_pattern = r'https?://(?:www\.)?youtube\.com/watch\?v=[0-9A-Za-z_-]{11}|https?://youtu\.be/[0-9A-Za-z_-]{11}'
        search_query = re.sub(url_pattern, '', question_lower)

        filler_phrases = presentation_keywords + recommendation_keywords + ["can you", "i want to learn", "give me", "article about", "about", "some", "best", "please", "based on the course materials", "based on", "from the materials", "summarize", "explain", "تلخيص", "شرح", "وضوح", "رشح", "ممكن", "عايز", "اتعلم", "عن", "افضل", "أفضل", "لي", "اعطني"]
        for phrase in filler_phrases:
            search_query = search_query.replace(phrase, "")
        search_query = " ".join(search_query.split()).strip(" ?!.،؟")

        if not search_query or len(search_query) < 2:
            search_query = re.sub(url_pattern, '', question).strip()
            if not search_query:
                search_query = "general"

        # -----------------------------------------------------------------
        # YouTube Processing
        # -----------------------------------------------------------------
        youtube_data = self.youtube_processor.process_url(question)
        youtube_transcript = youtube_data.get("transcript") if youtube_data else None
        video_meta = {"title": youtube_data.get("title"), "duration": youtube_data.get("duration"), "video_id": youtube_data.get("video_id")} if youtube_data else None

        # -----------------------------------------------------------------
        # ★ FAST TRACK: البريزنتيشن ★
        # -----------------------------------------------------------------
        if is_presentation:
            logger.info("Presentation intent detected...")
            raw_documents = forced_documents if forced_documents else self.retriever.retrieve(search_query)
            
            context_parts = []
            if youtube_data:
                meta_header = f"[VIDEO_TITLE: {video_meta['title']}]\n[VIDEO_DURATION: {video_meta['duration']}]\n"
                raw_transcript = youtube_data.get("transcript")
                content = raw_transcript if raw_transcript and "[ERROR:" not in str(raw_transcript) else "[No Transcript Available]"
                context_parts.append("[SOURCE: YOUTUBE_VIDEO_TRANSCRIPT]\n" + meta_header + content)
            
            if raw_documents:
                course_text = "\n\n".join([f"[Chunk {i+1}]:\n{doc.page_content}" for i, doc in enumerate(raw_documents)])
                context_parts.append("[SOURCE: OFFICIAL_COURSE_MATERIALS]\n" + course_text)
            
            full_context = "\n\n---\n\n".join(context_parts) if context_parts else question
            slides_data = self.generator.get_presentation_structure(full_context)

            user_images = [p for p in (image_paths or []) if p and os.path.exists(p)]
            pptx_path = self.presentation_maker.create_presentation(slides_data, user_images, "generated_presentation.pptx")

            if pptx_path:
                msg = f"لقد قمت بإنشاء العرض التقديمي لك. يمكنك العثور عليه هنا: {pptx_path}" if has_arabic else f"I have created a presentation for you. You can find it here: {pptx_path}"
                return {"answer": msg, "sources": [], "presentation_path": pptx_path}
            else:
                msg = "حاولت إنشاء عرض تقديمي ولكن حدث خطأ ما." if has_arabic else "I tried to create a presentation but something went wrong."
                return {"answer": msg, "sources": []}

        # -----------------------------------------------------------------
        # ★ FAST TRACK: الترشيحات ★
        # -----------------------------------------------------------------
        if is_recommendation:
            logger.info("Recommendation intent detected...")
            rec_query = search_query
            if (not rec_query or rec_query == "general") and video_meta and video_meta['title'] != "Unknown Title":
                rec_query = video_meta['title']
            recommendation_data = self.recommender.get_all_recommendations(rec_query)
            answer = self.generator.generate_answer(question, "", is_youtube=bool(youtube_data), history=history, recommendations=recommendation_data)
            return {"answer": answer, "sources": []}


        # ==================================================================
        # ★ AGENTIC TRACK: سؤال عادي ★
        # ==================================================================

        # 1. Agent Memory: Query Rewriting
        logger.info("Agent 1 (Memory): Rewriting query...")
        rewritten_query = self.generator.rewrite_query_with_memory(question, history or [])
        logger.info(f"Rewritten Query: {rewritten_query}")

        # -----------------------------------------------------------------
        # ★ 2. Smart Filtering Logic ★
        # -----------------------------------------------------------------
        active_filter = None
        if selected_course:
            logger.info(f"UI Filter ON: Searching ONLY in '{selected_course}'")
            active_filter = [selected_course.lower().strip()]
        elif user_courses:
            logger.info(f"UI Filter OFF: Searching in ALL enrolled courses: {user_courses}")
            active_filter = [c.lower().strip() for c in user_courses]
        else:
            logger.info("No courses provided at all. Searching globally.")
        # -----------------------------------------------------------------

        # 3. Agent Retrieval (First Pass - Enrolled Courses)
        logger.info("Agent 2 (Retriever): Fetching docs...")
        if forced_documents:
            documents = forced_documents
        else:
            documents = self.retriever.retrieve(rewritten_query, user_courses=active_filter)

        # 4. Prepare YouTube Context if exists
        context_parts = []
        if youtube_data:
            meta_header = f"[VIDEO_TITLE: {video_meta['title']}]\n[VIDEO_DURATION: {video_meta['duration']}]\n"
            raw_transcript = youtube_data.get("transcript")
            content = raw_transcript if raw_transcript and "[ERROR:" not in str(raw_transcript) else "[No Transcript Available]"
            context_parts.append("[SOURCE: YOUTUBE_VIDEO_TRANSCRIPT]\n" + meta_header + content)

        # -----------------------------------------------------------------
        # 5. لو مفيش مستندات من المحاضرات
        # -----------------------------------------------------------------
        if not documents:
            logger.warning("No documents retrieved from vector store.")
            route = self.generator.route_query(question)
            if route == "college_specific":
                ans = "عذراً، لم أجد هذه المعلومات في المحاضرات المسجلة عليك. يرجى التواصل مع قسم الكلية أو السكرتارية للتأكد." if has_arabic else "Sorry, I couldn't find this info in your enrolled courses. Please contact your department."
                return {"answer": ans, "sources": []}
            else:
                if context_parts:
                    full_context = "\n\n---\n\n".join(context_parts)
                    return {"answer": self.generator.generate_answer(question, full_context, is_youtube=True, history=history), "sources": []}
                ans = self.generator.generate_general_answer(question, history)
                return {"answer": ans, "sources": []}

        # -----------------------------------------------------------------
        # 6. تجهيز Context من المحاضرات
        # -----------------------------------------------------------------
        course_text = "\n\n".join([
            f"[Chunk {i+1} | File: {doc.metadata.get('file_name', 'unknown')}]:\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        context_parts.append("[SOURCE: OFFICIAL_COURSE_MATERIALS]\n" + course_text)
        full_context = "\n\n" + "\n\n---\n\n".join(context_parts)

        # -----------------------------------------------------------------
        # 7. Agent Evaluator: هل الداتا دي صح ولا False Positive؟
        # -----------------------------------------------------------------
        logger.info("Agent 3 (Evaluator): Checking if docs contain the answer...")
        evaluation = self.generator.evaluate_documents(rewritten_query, course_text) 
        logger.info(f"Evaluation Result: {evaluation}")

        # -----------------------------------------------------------------
        # 8. Agent Routing & Generation
        # -----------------------------------------------------------------
        if evaluation == "Yes":
            # الحالة الآمنة: الداتا صحيحة
            logger.info("Routing to Course Materials Generator...")
            answer = self.generator.generate_answer(
                question, full_context, is_youtube=bool(youtube_data), history=history
            )
            
            # ★ إزالة تكرار المصادر (نفس الملف ونفس الصفحة) ★
            seen_sources = set()
            unique_sources = []
            for doc in documents:
                file_name = doc.metadata.get('file_name', 'unknown')
                page = doc.metadata.get('page', 'unknown')
                source_key = f"{file_name}_p{page}"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    unique_sources.append({
                        "content": doc.page_content[:200] + "...", 
                        "metadata": doc.metadata
                    })
            
            return {"answer": answer, "sources": unique_sources}

        else:
            # ==================================================================
            # ★ الحالة الحرجة: الداتا مش كفاية (False Positive / Not Found)
            # ==================================================================
            
            # ★ هل نعمل محاولة ثانية في باقي الكلية؟ ★
            should_search_globally = (selected_course is None)
            
            if should_search_globally:
                logger.info("Not found in enrolled courses. Agent 3.5: Trying Global Search (Second Pass)...")
                global_docs = self.retriever.retrieve(rewritten_query, user_courses=None)
                
                if global_docs:
                    global_course_text = "\n\n".join([
                        f"[Chunk {i+1} | File: {doc.metadata.get('file_name', 'unknown')}]:\n{doc.page_content}"
                        for i, doc in enumerate(global_docs)
                    ])
                    global_eval = self.generator.evaluate_documents(rewritten_query, global_course_text)
                    
                    if global_eval == "Yes":
                        logger.info("Global Search succeeded! Found answer in other university courses.")
                        
                        # تجهيز الـ Context للإجابة الشاملة
                        global_context_parts = []
                        if youtube_data:
                            meta_header = f"[VIDEO_TITLE: {video_meta['title']}]\n[VIDEO_DURATION: {video_meta['duration']}]\n"
                            raw_transcript = youtube_data.get("transcript")
                            content = raw_transcript if raw_transcript and "[ERROR:" not in str(raw_transcript) else "[No Transcript Available]"
                            global_context_parts.append("[SOURCE: YOUTUBE_VIDEO_TRANSCRIPT]\n" + meta_header + content)

                        global_context_parts.append("[SOURCE: OFFICIAL_COURSE_MATERIALS]\n" + global_course_text)
                        full_global_context = "\n\n---\n\n".join(global_context_parts)
                        
                        # رسالة تنبيه للطالب
                        sys_warning = "\n\nSystem Note to Student: This answer was found in another course in your faculty's materials, not specifically in your enrolled courses." if not has_arabic else "\n\nملاحظة النظام: تم العثور على هذه الإجابة في مواد أخرى لكلية، وليس في المواد المسجلة لديك."
                        
                        answer = self.generator.generate_answer(
                            question, full_global_context, is_youtube=bool(youtube_data), history=history
                        ) + sys_warning
                        
                        # ★ إزالة تكرار المصادر للبحث الشامل ★
                        seen_global_sources = set()
                        global_sources = []
                        for doc in global_docs:
                            file_name = doc.metadata.get('file_name', 'unknown')
                            page = doc.metadata.get('page', 'unknown')
                            source_key = f"{file_name}_p{page}"
                            if source_key not in seen_global_sources:
                                seen_global_sources.add(source_key)
                                global_sources.append({
                                    "content": doc.page_content[:200] + "...", 
                                    "metadata": doc.metadata
                                })
                        
                        return {"answer": answer, "sources": global_sources}

            # ==================================================================
            # إذا وصلنا هنا: (البحث الشامل فشل) أو (اليوزر كان محدد مادة معينة ولازم نحترم اختياره)
            # ==================================================================
            logger.info("Routing to Fallback Logic (Global search failed or UI locked to specific course)...")
            route = self.generator.route_query(question)
            
            if route == "college_specific":
                ans = "عذراً، المعلومات المسترجعة لا تحتوي على إجابة دقيقة لسؤالك المتعلق بالجامعة. يرجى التواصل مع القسم المختص." if has_arabic else "Sorry, the retrieved documents do not contain a precise answer to your university-related question. Please contact the relevant department."
                return {"answer": ans, "sources": []}
            
            else:
                # سؤال عام، والمحاضرات مشتغطيه
                if youtube_data and youtube_transcript and "[ERROR:" not in str(youtube_transcript):
                    logger.info("Falling back to YouTube Transcript...")
                    yt_context = context_parts[0] if context_parts else ""
                    answer = self.generator.generate_answer(question, yt_context, is_youtube=True, history=history)
                    return {"answer": answer, "sources": []}

                logger.info("Routing to General Knowledge Generator...")
                ans = self.generator.generate_general_answer(question, history)
                return {"answer": ans, "sources": []}