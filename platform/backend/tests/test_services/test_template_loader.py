"""Tests for builtin template loader."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.test_template import TestTemplate
from app.utils.template_loader import load_builtin_templates


@pytest.mark.asyncio
async def test_load_builtin_templates_success(db_session: AsyncSession) -> None:
    """Test that builtin templates are loaded correctly."""
    # Run the loader
    await load_builtin_templates(db_session)

    # Verify templates were inserted
    query = select(TestTemplate).where(TestTemplate.is_builtin)
    result = await db_session.execute(query)
    templates = result.scalars().all()

    assert len(templates) > 0
    # Check for some specific template we know is in builtin_templates.json
    names = [t.name for t in templates]
    assert "Factual Recall" in names
    assert "Multi-hop Reasoning" in names


@pytest.mark.asyncio
async def test_load_builtin_templates_no_duplicates(db_session: AsyncSession) -> None:
    """Test that loading multiple times doesn't create duplicates."""
    # Run first time
    await load_builtin_templates(db_session)

    query = select(TestTemplate).where(TestTemplate.is_builtin)
    result = await db_session.execute(query)
    count1 = len(result.scalars().all())

    # Run second time
    await load_builtin_templates(db_session)

    result = await db_session.execute(query)
    count2 = len(result.scalars().all())

    assert count1 == count2
    assert count1 > 0
