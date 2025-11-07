# pip install playwright aiohttp pandas
# playwright install
import asyncio
import re
import aiohttp
import pandas as pd
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError, Page, BrowserContext
from bs4 import BeautifulSoup

URL = "https://gloton.ugr.es/microlensing/"

# Extended timeouts for computational jobs
NAV_TIMEOUT_MS = 180_000
LENS_GENERATION_TIMEOUT_MS = 10 * 60_000  # 10 minutes for lens generation
MAP_GENERATION_TIMEOUT_MS = 30 * 60_000   # 30 minutes for map generation
POLLING_INTERVAL_MS = 5_000
MAX_POLLING_ATTEMPTS = 360  # 30 minutes of polling

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

async def safe_click(page: Page, selector: str, timeout: int = 10_000) -> bool:
    """Safely click an element with multiple fallback strategies"""
    try:
        locator = page.locator(selector).first
        if await locator.count() > 0:
            await locator.scroll_into_view_if_needed(timeout=timeout)
            await locator.click(timeout=timeout)
            return True
    except Exception as e:
        log(f"Click failed for {selector}: {e}")
    return False

async def try_selectors(page: Page, selectors: list, wait_enabled: bool = False) -> bool:
    """Try multiple selectors and click the first available"""
    for sel in selectors:
        try:
            if wait_enabled:
                await page.wait_for_selector(f"{sel}:not([disabled])", timeout=5_000)
            if await safe_click(page, sel):
                return True
        except:
            continue
    return False

async def set_parameters(page: Page, values: list):
    """Set input parameters in Step 1"""
    log(f"Setting parameters: {values}")
    
    # Find all number inputs in Step 1 section
    inputs = page.locator('input[type="number"]')
    count = await inputs.count()
    log(f"Found {count} number inputs")
    
    for i, val in enumerate(values[:count]):
        try:
            input_field = inputs.nth(i)
            await input_field.scroll_into_view_if_needed()
            await input_field.click()
            await input_field.press("Control+A")
            
            # Format: integers for npix (last value), floats for others
            formatted_val = str(int(val)) if i == 7 else str(val)
            await input_field.fill(formatted_val)
            await asyncio.sleep(0.1)
        except Exception as e:
            log(f"Error setting parameter {i}: {e}")

async def wait_for_computation(page: Page, indicator_text: str, timeout_ms: int) -> bool:
    """Wait for a computation to complete by monitoring page content"""
    start = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - start) * 1000 < timeout_ms:
        try:
            content = await page.content()
            text = await page.evaluate("document.body.innerText")
            
            # Check for completion indicators
            if indicator_text.lower() in text.lower():
                return True
            
            # Check for error messages
            if "error" in text.lower() and "ready" not in text.lower():
                log(f"Possible error detected: {text[:200]}")
                
        except Exception as e:
            log(f"Error checking page content: {e}")
        
        await asyncio.sleep(2)
    
    return False

def lens_block(txt: str) -> Optional[float]:
    """
    Prints the 'Lens plane ready … Estimated exec time = X s' block if present,
    and returns the Estimated exec time as a float (in seconds).

    Parameters
    ----------
    txt : str
        Full page text to scan.

    Returns
    -------
    Optional[float]
        The extracted execution time in seconds, or None if not found.
    """
    # Capture the full block
    block_match = re.search(
        r"(Lens plane ready.*?Estimated exec time\s*=\s*[-+0-9.,Ee]+?\s*s\.?)",
        txt, flags=re.IGNORECASE | re.DOTALL
    )

    # Capture just the numeric value of 'Estimated exec time'
    time_match = re.search(
        r"Estimated exec time\s*=\s*([-+0-9.,Ee]+)", txt,
        flags=re.IGNORECASE
    )

    if block_match:
        # Print cleaned block
        print("\n".join(
            line.strip() for line in block_match.group(1).splitlines()
            if line.strip()
        ), flush=True)

    # Return float if available
    if time_match:
        try:
            return float(time_match.group(1).replace(",", ""))
        except ValueError:
            pass
    return 0

async def page_text(p: Page) -> str:
    return await p.evaluate("() => document.body ? document.body.innerText : ''")


async def extract_outputs_url(page: Page) -> Optional[str]:
    """Extract outputs URL from page using multiple methods"""
    
    # Method 1: Check all links
    try:
        links = await page.evaluate("""
            () => Array.from(document.querySelectorAll('a[href]'))
                .map(a => a.href)
                .filter(href => href.includes('/outputs/'))
        """)
        if links:
            log(f"Found outputs links: {links}")
            return links[0]
    except Exception as e:
        log(f"Method 1 failed: {e}")
    
    # Method 2: Check page text for URL pattern
    try:
        text = await page.evaluate("document.body.innerText")
        match = re.search(r'https?://[^\s<>"]+/outputs/[^\s<>"]+', text, re.I)
        if match:
            url = match.group(0)
            log(f"Found outputs URL in text: {url}")
            return url
    except Exception as e:
        log(f"Method 2 failed: {e}")
    
    # Method 3: Check for job ID and construct URL
    try:
        text = await page.evaluate("document.body.innerText")
        job_match = re.search(r'job[_-]?id[:\s]+([a-zA-Z0-9_-]+)', text, re.I)
        if job_match:
            job_id = job_match.group(1)
            constructed_url = f"https://gloton.ugr.es/microlensing/outputs/{job_id}/index.html"
            log(f"Constructed outputs URL: {constructed_url}")
            return constructed_url
    except Exception as e:
        log(f"Method 3 failed: {e}")
    
    return None

async def download_file(url: str, output_path: Path) -> bool:
    """Download file using aiohttp"""
    log(f"Downloading {url}...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'wb') as f:
                        f.write(await response.read())
                    log(f"✓ Downloaded to {output_path}")
                    return True
                else:
                    log(f"✗ Download failed with status {response.status}")
                    return False
    except Exception as e:
        log(f"✗ Download error: {e}")
        return False

async def run_microlensing_simulation(
    values: List[float],
    output_dir: Path,
    row_id: int = 0,
    headless: bool = True
) -> dict:
    """
    Run a single microlensing simulation with given parameters
    
    Args:
        values: List of 8 parameters [kappa, gamma, s, mmin, mmax, beta, xl, npix]
        output_dir: Directory to save output files
        row_id: Identifier for this simulation (e.g., DataFrame row index)
        headless: Run browser in headless mode
    
    Returns:
        dict with status and file paths
    """
    result = {
        'row_id': row_id,
        'values': values,
        'status': 'failed',
        'dat_gz_path': None,
        'magmap_path': None,
        'error': None
    }
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        
        page = await context.new_page()
        page.set_default_timeout(NAV_TIMEOUT_MS)
        
        try:
            log(f"{'='*60}")
            log(f"Starting simulation {row_id}")
            log(f"{'='*60}")
            
            # Step 1: Navigate to site
            log("Opening site...")
            await page.goto(URL, wait_until="domcontentloaded")
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(2)
            
            # Accept cookies
            cookie_selectors = [
                'button:has-text("Accept")',
                'button:has-text("Aceptar")',
                '#cookie-accept',
                '.cookie-accept'
            ]
            await try_selectors(page, cookie_selectors)
            
            # Step 2: Set parameters
            await set_parameters(page, values)
            await asyncio.sleep(1)
            
            # Apply parameters
            apply_selectors = [
                'button:has-text("Apply")',
                'button:has-text("Aplicar")',
                'input[type="button"][value*="Apply"]'
            ]
            if await try_selectors(page, apply_selectors):
                log("Parameters applied")
                await asyncio.sleep(2)
            
            # Step 3: Generate lens
            log("Generating lens...")
            lens_selectors = [
                '#calclens',
                'button:has-text("Generate lens")',
                'button:has-text("Generar lente")'
            ]
            
            # Wait before clicking to ensure page is fully ready
            log("Waiting 5 seconds before clicking Generate Lens...")
            await asyncio.sleep(5)
            
            if await try_selectors(page, lens_selectors, wait_enabled=True):
                log("Lens generation started")
                
                # Wait for lens to be ready
                if await wait_for_computation(page, "Lens plane ready", LENS_GENERATION_TIMEOUT_MS):
                    log("✓ Lens generation completed")
                else:
                    log("⚠ Lens generation timeout - continuing anyway")
            
            await asyncio.sleep(3)
            
            # Step 4: Generate map
            log("Generating map...")
            map_selectors = [
                '#calcmap',
                'button:has-text("Generate map")',
                'button:has-text("Generar mapa")'
            ]
            
            if await try_selectors(page, map_selectors, wait_enabled=True):
                log("Map generation started - this may take 10-30 minutes...")
                await asyncio.sleep(5)
            
            # Step 5: Wait for and extract outputs URL
            log("Waiting for outputs URL...")
            outputs_url = None
            
            for attempt in range(MAX_POLLING_ATTEMPTS):
                outputs_url = await extract_outputs_url(page)
                if outputs_url:
                    break
                
                if attempt % 12 == 0:  # Every minute
                    log(f"Still waiting... (attempt {attempt + 1}/{MAX_POLLING_ATTEMPTS})")
                
                await asyncio.sleep(POLLING_INTERVAL_MS / 1000)
            
            if not outputs_url:
                log("✗ Failed to find outputs URL")
                result['error'] = "Outputs URL not found"
                
                # Dump page content for debugging
                content = await page.content()
                debug_path = output_dir / f"debug_page_{row_id}.html"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, "w") as f:
                    f.write(content)
                log(f"Page content saved to {debug_path}")
                log(content)
                return result
            
            rest_time = lens_block(await page_text(page)) *1.5
            
            
            # Step 6: Navigate to outputs page
            log(f"Found outputs URL: {outputs_url}")
            log(f"Waiting {rest_time} seconds before navigating to outputs page...")
            await asyncio.sleep(rest_time)
            log(f"Now opening outputs page: {outputs_url}")
            await page.goto(outputs_url, wait_until="domcontentloaded")
            await page.wait_for_load_state("networkidle")
            
            # Step 7: Poll for .dat.gz file
            log("Polling for .dat.gz file...")
            dat_gz_url = None
            
            for attempt in range(MAX_POLLING_ATTEMPTS):
                try:
                    # Check for download link
                    dat_links = await page.evaluate("""
                        () => Array.from(document.querySelectorAll('a[href$=".dat.gz"]'))
                            .map(a => a.href)
                    """)
                    
                    if dat_links:
                        dat_gz_url = dat_links[0]
                        log(f"✓ Found .dat.gz file: {dat_gz_url}")
                        break
                    
                    # Check page content
                    text = await page.evaluate("document.body.innerText")
                    if "not finished" not in text.lower() and "processing" not in text.lower():
                        # May be ready but link not found - save for debugging
                        if attempt > 10:
                            debug_path = output_dir / f"outputs_page_{row_id}.html"
                            with open(debug_path, "w") as f:
                                f.write(await page.content())
                    
                    if attempt % 12 == 0:
                        log(f"Still polling outputs page... (attempt {attempt + 1}/{MAX_POLLING_ATTEMPTS})")
                    
                    await asyncio.sleep(POLLING_INTERVAL_MS / 1000)
                    await page.reload(wait_until="domcontentloaded")
                    
                except Exception as e:
                    log(f"Error during polling: {e}")
                    await asyncio.sleep(POLLING_INTERVAL_MS / 1000)
            
            if not dat_gz_url:
                log("✗ .dat.gz file never became available")
                result['error'] = ".dat.gz file not available"
                return result
            
            # Step 8: Download the main data file
            output_file = output_dir / f"data.dat.gz"
            if await download_file(dat_gz_url, output_file):
                result['dat_gz_path'] = str(output_file)
                log(f"✓ Main file downloaded to {output_file}")
            
            # Step 9: Also download magmap.dat.gz if available
            try:
                # Extract job_id from outputs_url to construct magmap URL
                job_id_match = re.search(r'/outputs/([^/]+)/', outputs_url)
                if job_id_match:
                    job_id = job_id_match.group(1)
                    magmap_url = f"https://gloton.ugr.es/microlensing/outputs/{job_id}/magmap.dat.gz"
                    log(f"Attempting to download magmap.dat.gz from {magmap_url}")
                    
                    magmap_file = output_dir / f"magmap.dat.gz"
                    if await download_file(magmap_url, magmap_file):
                        result['magmap_path'] = str(magmap_file)
                        log(f"✓ magmap.dat.gz downloaded to {magmap_file}")
                    else:
                        log("⚠ magmap.dat.gz download failed or not available")
                else:
                    log("⚠ Could not extract job_id for magmap download")
            except Exception as e:
                log(f"⚠ Error downloading magmap.dat.gz: {e}")
            
            result['status'] = 'success'
            log(f"✓ Simulation {row_id} completed successfully!")
            
        except Exception as e:
            log(f"✗ Fatal error in simulation {row_id}: {e}")
            result['error'] = str(e)
            import traceback
            traceback.print_exc()
            
        finally:
            await context.close()
            await browser.close()
    
    return result

async def run_batch_simulations(
    df: pd.DataFrame,
    param_columns: List[str],
    output_base_dir: str = "./microlensing_outputs",
    headless: bool = True,
    max_concurrent: int = 1
):
    """
    Run multiple microlensing simulations from a DataFrame
    
    Args:
        df: DataFrame with parameters
        param_columns: List of column names for the 8 parameters
        output_base_dir: Base directory for all outputs
        headless: Run browsers in headless mode
        max_concurrent: Maximum number of concurrent simulations (recommend 1-2)
    
    Returns:
        DataFrame with results
    """
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save input parameters
    params_file = output_dir / "input_parameters.csv"
    df.to_csv(params_file, index=False)
    log(f"Input parameters saved to {params_file}")
    
    results = []
    
    # Process simulations (sequential or with limited concurrency)
    if max_concurrent == 1:
        # Sequential processing
        for idx, row in df.iterrows():
            path_obj = output_dir / (row["name_obj"] + row["image"])
            path_obj.mkdir(parents=True, exist_ok=True)
            values = [row[col] for col in param_columns]
            result = await run_microlensing_simulation(
                values=values,
                output_dir= path_obj,
                row_id=idx,
                headless=headless
            )
            results.append(result)
            
            # Save intermediate results after each simulation
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_dir / "results.csv", index=False)
            log(f"Progress: {len(results)}/{len(df)} simulations completed")
    else:
        # Concurrent processing with semaphore
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(idx, row):
            async with semaphore:
                values = [row[col] for col in param_columns]
                return await run_microlensing_simulation(
                    values=values,
                    output_dir= path_obj,
                    row_id=idx,
                    headless=headless
                )
        
        tasks = [run_with_semaphore(idx, row) for idx, row in df.iterrows()]
        results = await asyncio.gather(*tasks)
    
    # Final results
    results_df = pd.DataFrame(results)
    results_file = output_dir / "final_results.csv"
    results_df.to_csv(results_file, index=False)
    
    log(f"\n{'='*60}")
    log(f"BATCH PROCESSING COMPLETE")
    log(f"{'='*60}")
    log(f"Total simulations: {len(results)}")
    log(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    log(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    log(f"Results saved to: {results_file}")
    
    return results_df

# Example usage
if __name__ == "__main__":
    # Example 1: Single simulation
    # print("Example 1: Single simulation")
    # print("-" * 60)
    
    # single_values = [0.6946495, 0.6945083, 0.2, 0.299, 0.3, 0.01, 30, 1100]
    # output_dir = Path("./single_simulation_output")
    
    # result = asyncio.run(run_microlensing_simulation(
    #     values=single_values,
    #     output_dir=output_dir,
    #     row_id=0,
    #     headless=False  # Set to True for production
    # ))
    
    # print("\nResult:", result)
    
    # Example 2: Batch processing from DataFrame
    print("\n\n Batch processing")
    print("-" * 60)
    
    # Create sample DataFrame with multiple parameter sets
    df = pd.read_csv("jjv_params.csv")#.iloc[67:]
    df['Smooth_matter_fraction(alpha)'] = 0.4
    param_cols = ['Convergence', 'Shear', 'Smooth_matter_fraction(alpha)', 'Minmass', 'Maxmass', 'Lens_mass_function_power', 'Mapsize(Er)', 'Numberofpixels']
    
    results_df = asyncio.run(run_batch_simulations(
        df=df,
        param_columns=param_cols,
        output_base_dir="./Mapas_04",
        headless=False,  # Set to True for production
        max_concurrent=1  # Process one at a time (recommended)
    ))
    
    print("\n\nFinal Results:")
    print(results_df)