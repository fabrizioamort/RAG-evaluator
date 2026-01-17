"""BUiltin template loader."""

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.test_template import TestTemplate
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


async def load_builtin_templates(db: AsyncSession) -> None:
    """Load builtin templates from JSON file into the database.

    This should be called on application startup. It only inserts templates
    that don't already exist in the database (matching by name).
    """
    templates_path = Path("data/templates/builtin_templates.json")
    if not templates_path.exists():
        logger.warning("Builtin templates file not found", path=str(templates_path))
        return

    try:
        with open(templates_path, encoding="utf-8") as f:
            builtin_templates = json.load(f)
    except Exception as e:
        logger.error("Failed to read builtin templates file", error=str(e))
        return

    count = 0
    for t_data in builtin_templates:
        # Check if template already exists by name
        query = select(TestTemplate).where(TestTemplate.name == t_data["name"])
        result = await db.execute(query)
        existing = result.scalar_one_or_none()

        if not existing:
            template = TestTemplate(
                name=t_data["name"],
                description=t_data.get("description"),
                category=t_data.get("category"),
                question_template=t_data["question_template"],
                answer_template=t_data.get("answer_template"),
                entity_types=t_data.get("entity_types", []),
                complexity_level=t_data.get("complexity_level", "medium"),
                is_builtin=True,
            )
            db.add(template)
            count += 1
        else:
            # Optionally update existing builtin templates if needed
            # For now, we just skip
            pass

    if count > 0:
        await db.commit()
        logger.info("Loaded builtin templates", count=count)
    else:
        logger.info("No new builtin templates to load")
