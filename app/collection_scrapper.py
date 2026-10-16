"""
Tool to build chromaDB scrapping local music collection
"""
import logging
from core.chroma_manager import chroma_service

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to initialize and run the local collection scrapper"""
    logger.info("Initializing local collection scrapping")
    
    try:
        # Check ChromaDB context
        chroma_service.scrap_and_process_collection()

        logger.info(f"Chroma collection updated successfully!")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()