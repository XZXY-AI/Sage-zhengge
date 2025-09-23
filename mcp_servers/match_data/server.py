# 1. 导入所有必要的库
from datetime import datetime, timedelta
import re

import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount
import uvicorn
import argparse
from typing import List, Dict, Any, Optional, Union
import traceback  # <-- 导入 traceback 模块用于打印详细错误
from rank_bm25 import BM25Okapi

"""
你看一下，为了更好的分析赛事
你看还有哪里可以优化的,目的就是为了让使用者一句话可以得到一个赛事分析
"""

# 2. 初始化 MCP 框架 (不变)
mcp = FastMCP("Match Data Service")

# 3. 设置命令行参数 (不变)
parser = argparse.ArgumentParser(description='启动比赛数据 MCP Server')
args = parser.parse_args()


# ==============================================================================

# query_match_list_by_date 工具获取今天19点35，天津津门虎vs成都蓉城的内容，比赛状态现在是未开始， 我需要获取到这次的赛事id 不要使用其他mcp工具
# 请调用 query_match_list_by_date 工具获取今天19点35，天津津门虎vs成都蓉城的内容，比赛状态现在是未开始， 我需要获取到这次的赛事id
# 请调用 query_match_list_by_date 工具获取今天的足球比赛列表。然后，仅提取每场比赛的主队名称、客队名称、联赛名称和比赛，并以列表的形式呈现给我。


@mcp.tool()
async def query_match_list_with_filter(
        match_type: str,
        date: Optional[str] = None,
        team_name_query: Optional[str] = None,
        league_ids: Optional[str] = None,
        team_id: Optional[str] = None,
        status: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """
    根据指定条件筛选获取比赛列表，支持BM25相似度查询和数量限制。

    Args:
        match_type (str): 体育类型，必需参数。1 代表足球, 2 代表篮球。
        date (Optional[str]): 查询日期，格式为 YYYY-MM-DD。
        team_name_query (Optional[str]): 队伍以及赛事名称相关内容查询，将与主队、客队和联赛名称进行BM25相似度匹配。 
        league_ids (Optional[str]): 联赛ID，多个ID用逗号分隔。
        team_id (Optional[str]): 根据球队ID筛选比赛。
        status (Optional[str]): 根据比赛状态筛选比赛。1:未开始, 2:完场, 3:赛中, 4:其他。
    
    Returns:
        Dict containing:
        - matches: 最多10场比赛的详细信息（如有team_name_query则按相似度排序）
        - statistics: 全局比赛统计信息
    """
    params = {"match_type": "1"}
    endpoint = "http://ai-match.fengkuangtiyu.cn/api/v5/matches"
    if match_type:
        params["match_type"] = match_type
    if date:
        cleaned_date = date.replace(" ", "")
        params["date"] = cleaned_date
    if league_ids:
        params["league_ids"] = league_ids
    if team_id:
        params["team_id"] = team_id
    if status:
        params["status"] = status
    print(f"Executing get_match_list with params: {params}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()

            data = response.json()
            # Check if the API call was successful
            if data.get("code") == "0":
                all_matches = data.get("data", [])
                
                # 计算全局统计信息
                total_matches = len(all_matches)
                statistics = {
                    "total_matches": total_matches
                }
                
                # 使用BM25对比赛进行排序（如果有查询则排序，没有查询则保持原顺序）
                if team_name_query and all_matches:
                    print(f"Applying BM25 similarity search for query: {team_name_query}")
                    sorted_matches = sort_matches_with_bm25(team_name_query, all_matches)
                else:
                    sorted_matches = all_matches
                
                # 取前20个比赛
                limited_matches = sorted_matches[:10]
                statistics["returned_matches"] = len(limited_matches)
                
                # 添加状态分布统计
                if all_matches:
                    status_counts = {}
                    league_counts = {}
                    
                    for match in all_matches:
                        # 统计状态分布
                        match_status = match.get('status', 'unknown')
                        status_counts[match_status] = status_counts.get(match_status, 0) + 1
                        
                        # 统计联赛分布
                        league_name = match.get('league_name', 'unknown')
                        league_counts[league_name] = league_counts.get(league_name, 0) + 1
                    
                    statistics["status_distribution"] = status_counts
                    statistics["top_leagues"] = dict(sorted(league_counts.items(), 
                                                          key=lambda x: x[1], reverse=True)[:10])
                
                return {
                    "matches": limited_matches,
                    "statistics": statistics
                }
            else:
                # If the API returned an error, fulfill the promise by returning a STRING
                return f"API returned an error: {data.get('msg', 'Unknown error')}"

    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        print(f"--- DETAILED ERROR IN query_match_list_by_date ---")
        traceback.print_exc()
        return f"An unexpected error occurred in query_match_list_by_date: [Type: {type(e).__name__}] - [Details: {repr(e)}]"

# 获取比赛联赛排名和积分
# http://ai-match.fengkuangtiyu.cn/api/v5/matches/3558764/league_standings?match_type=1
@mcp.tool()
async def get_match_standings_by_id(
        match_id: str,
        match_type: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """
    根据比赛获取联赛排名和积分。
    Fetches league standings and points by match ID.

    Args:
        match_id (str): 比赛 (必需), 例如 '3558764'。
        match_type (Optional[str]): 体育类型, 1 代表足球, 2 代表篮球。

    Returns:
        Union[Dict[str, Any], str]: 成功时返回排名积分数据，失败时返回错误信息字符串。
    """
    endpoint = f"http://ai-match.fengkuangtiyu.cn/api/v5/matches/{match_id}/league_standings"

    # --- vvv 修改的部分 vvv ---
    # 即使 AI 没有提供 match_type，我们也默认设置为 '1'
    params = {"match_type": "1"}
    # 如果 AI 提供了，就使用 AI 提供的值覆盖默认值
    if match_type:
        params["match_type"] = match_type
    # --- ^^^ 修改的部分 ^^^ ---

    print(f"Executing get_league_standings for match_id: {match_id} with params: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_league_standings ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_league_standings: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


# 辅助函数：递归移除字典中的 logo_url 字段
def remove_logo_url_from_dict(data):
    """
    递归地从字典或列表中移除所有的 logo_url 字段
    
    Args:
        data: 要处理的数据（字典、列表或其他类型）
    
    Returns:
        处理后的数据，移除了所有 logo_url 字段
    """
    if isinstance(data, dict):
        # 创建新字典，排除 logo_url 字段
        result = {}
        for key, value in data.items():
            if key != "logo_url":
                result[key] = remove_logo_url_from_dict(value)
        return result
    elif isinstance(data, list):
        # 递归处理列表中的每个元素
        return [remove_logo_url_from_dict(item) for item in data]
    else:
        # 其他类型直接返回
        return data

# 比赛信息
# http://ai-match.fengkuangtiyu.cn/api/v5/matches/3558764/essentials?match_type=1
@mcp.tool()
async def get_match_details_by_id(
        match_id: str,
        match_type: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """
    根据比赛获取比赛的核心信息。
    Fetches the essential information for a match by its ID.

    Args:
        match_id (str): 比赛 (必需), 例如 '3558764'。
        match_type (Optional[str]): 体育类型, 1 代表足球, 2 代表篮球。

    Returns:
        Union[Dict[str, Any], str]: 成功时返回比赛信息数据，失败时返回错误信息字符串。
    """
    endpoint = f"http://ai-match.fengkuangtiyu.cn/api/v5/matches/{match_id}/essentials"

    # 即使 AI 没有提供 match_type，我们也默认设置为 '1'
    params = {"match_type": "1"}
    if match_type:
        params["match_type"] = match_type

    print(f"Executing get_match_essentials for match_id: {match_id} with params: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            original_data = response.json()
            
            # 移除所有 logo_url 字段
            cleaned_data = remove_logo_url_from_dict(original_data)
            return cleaned_data
            
    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_match_essentials ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_match_essentials: [Type: {type(e).__name__}] - [Details: {repr(e)}]"
# 请调用 get_team_recent_performance_by_match_id 工具，参数为 match_id: "3566777"，总结一下这个比赛的双方近期战绩结果。
import re
from datetime import datetime

# 辅助函数：解析日期字符串
def parse_date_string(date_str: str) -> datetime:
    """
    解析日期字符串，支持多种格式
    
    Args:
        date_str: 日期字符串，如 "05-12-3"、"2005-12-03" 或 "2025-09-24T01:30:00.000+00:00"
    
    Returns:
        datetime对象
    """
    if not date_str:
        return datetime.min
    
    try:
        # 处理 ISO 8601 格式 "2025-09-24T01:30:00.000+00:00"
        if 'T' in date_str:
            # 移除时区信息和毫秒，只保留日期时间部分
            date_part = date_str.split('T')[0]
            time_part = date_str.split('T')[1].split('.')[0] if '.' in date_str else date_str.split('T')[1].split('+')[0].split('Z')[0]
            datetime_str = f"{date_part} {time_part}"
            return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        
        # 处理 "05-12-3" 格式
        elif re.match(r'^\d{2}-\d{1,2}-\d{1,2}$', date_str):
            parts = date_str.split('-')
            year = int(f"20{parts[0]}")  # 05 -> 2005
            month = int(parts[1])
            day = int(parts[2])
            return datetime(year, month, day)
        
        # 处理 "2005-12-03" 格式
        elif re.match(r'^\d{4}-\d{1,2}-\d{1,2}$', date_str):
            parts = date_str.split('-')
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            return datetime(year, month, day)
        
        # 处理其他可能的格式
        else:
            # 尝试直接解析
            return datetime.strptime(date_str, "%Y-%m-%d")
            
    except (ValueError, IndexError) as e:
        print(f"DEBUG: 日期解析失败 - {date_str}, 错误: {e}")
        return datetime.min

# 辅助函数：对比赛列表按日期排序并取最近N场
def sort_and_limit_matches(matches: List[Dict], limit: int = 10, upcoming: bool = False) -> List[Dict]:
    """
    对比赛列表按日期排序并限制数量
    
    Args:
        matches: 比赛列表
        limit: 限制数量，默认10场
        upcoming: 是否为即将开始的比赛，True时按升序排序（最临近的在前），False时按降序排序（最新的在前）
    
    Returns:
        排序并限制后的比赛列表
    """
    if not matches or not isinstance(matches, list):
        return []
    
    # 为每场比赛添加解析后的日期用于排序
    matches_with_dates = []
    for match in matches:
        if isinstance(match, dict):
            date_str = match.get('date', '')
            parsed_date = parse_date_string(date_str)
            match_copy = match.copy()
            match_copy['_parsed_date'] = parsed_date
            matches_with_dates.append(match_copy)
    
    # 根据upcoming参数决定排序方向
    if upcoming:
        # 对于即将开始的比赛，按日期升序排序（最临近的在前）
        sorted_matches = sorted(matches_with_dates, key=lambda x: x['_parsed_date'], reverse=False)
    else:
        # 对于历史比赛，按日期降序排序（最新的在前）
        sorted_matches = sorted(matches_with_dates, key=lambda x: x['_parsed_date'], reverse=True)
    
    # 移除临时添加的日期字段并限制数量
    result = []
    for match in sorted_matches[:limit]:
        match_clean = match.copy()
        if '_parsed_date' in match_clean:
            del match_clean['_parsed_date']
        result.append(match_clean)
    
    return result

# 辅助函数：将比赛结果数字转换为语义可读的内容
def convert_result_to_readable(result: Union[int, str]) -> str:
    """
    将比赛结果数字转换为语义可读的内容
    
    Args:
        result: 比赛结果，0=负，1=平，3=胜
        
    Returns:
        语义可读的结果字符串
    """
    try:
        result_int = int(result)
        if result_int == 0:
            return "负"
        elif result_int == 1:
            return "平"
        elif result_int == 3:
            return "胜"
        else:
            return f"未知({result})"
    except (ValueError, TypeError):
        return f"无效({result})"


def convert_matches_result_to_readable(matches: List[Dict]) -> List[Dict]:
    """
    将比赛列表中的result字段转换为语义可读内容
    
    Args:
        matches: 比赛列表
        
    Returns:
        转换后的比赛列表
    """
    if not matches:
        return matches
    
    converted_matches = []
    for match in matches:
        # 创建副本避免修改原始数据
        match_copy = match.copy()
        if 'result' in match_copy:
            match_copy['result'] = convert_result_to_readable(match_copy['result'])
        converted_matches.append(match_copy)
    
    return converted_matches

# BM25相似度计算相关函数
def tokenize_text(text: str) -> List[str]:
    """
    文本分词，支持中英文
    
    Args:
        text: 输入文本
    
    Returns:
        分词结果列表
    """
    if not text:
        return []
    
    # 转换为小写并移除特殊字符
    text = text.lower()
    # 分离中英文和数字
    tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text)
    
    # 对中文进行字符级分割
    result = []
    for token in tokens:
        if re.match(r'[\u4e00-\u9fff]+', token):
            # 中文字符级分割
            result.extend(list(token))
        else:
            # 英文和数字保持完整
            result.append(token)
    
    return result

def sort_matches_with_bm25(query: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    使用BM25算法对比赛列表进行排序
    
    Args:
        query: 查询字符串
        matches: 比赛列表
    
    Returns:
        按相似度排序的比赛列表，输入多长输出多长，只是重新排序
    """
    if not matches:
        return []
    
    # 如果没有查询字符串，直接返回原始列表
    if not query or not query.strip():
        return matches
    
    # 构建文档语料库
    documents = []
    for match in matches:
        # 构建比赛文本：主队名 + 客队名 + 联赛名
        match_text_parts = []
        
        # 获取队伍名称
        teams = match.get('teams', {})
        if isinstance(teams, dict):
            home_team = teams.get('home', {})
            away_team = teams.get('away', {})
            
            if isinstance(home_team, dict) and 'name' in home_team:
                match_text_parts.append(home_team['name'])
            if isinstance(away_team, dict) and 'name' in away_team:
                match_text_parts.append(away_team['name'])
        
        # 获取联赛名称
        league_name = match.get('league_name', '')
        if league_name:
            match_text_parts.append(league_name)
        
        # 合并文本并分词（使用字符级分词）
        match_text = ' '.join(match_text_parts)
        tokenized_doc = tokenize_text(match_text)
        documents.append(tokenized_doc)
    
    # 如果没有有效文档，返回原始列表
    if not documents or all(not doc for doc in documents):
        return matches
    
    # 创建BM25模型
    bm25 = BM25Okapi(documents)
    
    # 查询分词（使用字符级分词）
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return matches
    
    # 获取相似度分数
    scores = bm25.get_scores(query_tokens)
    
    # 创建(分数, 索引, 比赛)的元组列表，包含所有比赛
    scored_matches = []
    for i, (score, match) in enumerate(zip(scores, matches)):
        scored_matches.append((score, i, match))
    
    # 按分数降序排序（相似度高的在前）
    scored_matches.sort(key=lambda x: x[0], reverse=True)
    
    # 返回所有排序后的比赛，输入多长输出多长
    return [match for _, _, match in scored_matches]

# 辅助函数：计算单队近10场战绩
def calculate_team_recent_10_games(matches: List[Dict], team_key: str = None) -> str:
    """
    计算队伍近10场比赛的胜负平统计
    
    Args:
        matches: 比赛列表
        team_key: 队伍标识，用于区分主客队（如果需要）
    
    Returns:
        格式化的战绩字符串，如 '8Win2Lose0Draw'
    """
    if not matches or not isinstance(matches, list):
        return "0Win0Lose0Draw"
    
    # 只取前10场比赛
    recent_matches = matches[:10]
    
    wins = 0
    loses = 0
    draws = 0
    
    for match in recent_matches:
        if not isinstance(match, dict):
            continue
            
        result = match.get('result')
        # 处理不同的字段名：penalty_score 或 penaltyScore
        penalty_score = match.get('penalty_score') or match.get('penaltyScore', '')
        # 处理不同的字段名：is_home 或 isHome
        is_home = match.get('is_home') or match.get('isHome')
        
        print(f"DEBUG: 处理比赛 - date: {match.get('date')}, result: {result}, penalty_score: {penalty_score}, is_home: {is_home}")
        
        # 确保result和is_home都是整数类型
        try:
            result = int(result) if result is not None else None
        except (ValueError, TypeError):
            result = None
            
        try:
            is_home = int(is_home) if is_home is not None else None
        except (ValueError, TypeError):
            is_home = None
            
        print(f"DEBUG: 转换后 - result: {result}, is_home: {is_home}")
            
        if result == 3:  # 胜
            wins += 1
            print(f"DEBUG: result=3, 胜")
        elif result == 0:  # 负
            loses += 1
            print(f"DEBUG: result=0, 负")
        elif result == 1:  # 需要进一步判断
            print(f"DEBUG: result=1, 检查点球")
            if not penalty_score:  # 没有点球，则是平局
                draws += 1
                print(f"DEBUG: 无点球, 平局")
            else:
                print(f"DEBUG: 有点球: {penalty_score}")
                # 有点球，根据is_home和penalty_score判断胜负
                # penalty_score格式可能是 "4:2" 或类似格式
                split_pattern = r'[\-–—−]'
                score_parts = re.split(split_pattern, penalty_score)
                if len(score_parts) == 2:  # 确保分割后正好是两部分
                    try:
                        left_score = int(score_parts[0].strip())
                        right_score = int(score_parts[1].strip())

                        print(
                            f"DEBUG: 点球处理 - date: {match.get('date')}, is_home: {is_home}, penalty: {penalty_score}, left: {left_score}, right: {right_score}")

                        if is_home == 1:
                            if left_score > right_score:
                                wins += 1
                                print(f"DEBUG: 主队点球胜")
                            else:
                                loses += 1
                                print(f"DEBUG: 主队点球负")
                        elif is_home == 0:
                            if right_score > left_score:
                                wins += 1
                                print(f"DEBUG: 客队点球胜")
                            else:
                                loses += 1
                                print(f"DEBUG: 客队点球负")
                        else:
                            loses += 1
                            print(f"DEBUG: is_home值无效，计为负")

                    except (ValueError, AttributeError) as e:
                        print(f"DEBUG: 点球解析异常 - {e}，计为负")
                        loses += 1
                else:
                    print(f"DEBUG: 点球格式分割后部分不为2，计为负")
                    loses += 1
        # 其他result值默认不统计
    
    return f"{wins}Win{loses}Lose{draws}Draw"


# http://ai-match.fengkuangtiyu.cn/api/v5/matches/3558764/recent_forms?match_type=1
# 获取比赛近期战绩
# 调用工具根据比赛 3558764获取他的近期战绩
@mcp.tool()
async def get_team_recent_performance_by_match_id(
        match_id: str,
        match_type: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """
    根据比赛获取比赛双方的近期战绩，自动按日期排序并取最近10场比赛。
    Fetches the recent forms of the teams involved in a match by its ID, automatically sorted by date and limited to the most recent 10 matches.

    Args:
        match_id (str): 比赛 (必需), 例如 '3558764'。
        match_type (Optional[str]): 体育类型, 1 代表足球, 2 代表篮球。

    Returns:
        Union[Dict[str, Any], str]: 成功时返回近期战绩数据（已按日期排序并限制为最近10场），失败时返回错误信息字符串。
    """
    endpoint = f"http://ai-match.fengkuangtiyu.cn/api/v5/matches/{match_id}/recent_forms"

    params = {"match_type": "1"}
    if match_type:
        params["match_type"] = match_type

    print(f"Executing get_match_recent_forms for match_id: {match_id} with params: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            original_data = response.json()
            
            # 如果API返回成功，处理数据排序和统计
            if isinstance(original_data, dict) and original_data.get("code") == "0":
                data = original_data.get("data", {})
                
                # 处理主队近期战绩 - 按日期排序并取最近10场
                if "home_team" in data and isinstance(data["home_team"], list):
                    home_matches = data["home_team"]
                    # 按日期排序并限制为最近10场比赛
                    sorted_home_matches = sort_and_limit_matches(home_matches, 10)
                    # 转换result字段为语义可读内容
                    readable_home_matches = convert_matches_result_to_readable(sorted_home_matches)
                    data["home_team"] = readable_home_matches
                    
                    # 计算战绩统计
                    home_recent_10 = calculate_team_recent_10_games(sorted_home_matches)
                    data["home_team_recent10Games"] = home_recent_10
                
                # 处理客队近期战绩 - 按日期排序并取最近10场
                if "guest_team" in data and isinstance(data["guest_team"], list):
                    guest_matches = data["guest_team"]
                    # 按日期排序并限制为最近10场比赛
                    sorted_guest_matches = sort_and_limit_matches(guest_matches, 10)
                    # 转换result字段为语义可读内容
                    readable_guest_matches = convert_matches_result_to_readable(sorted_guest_matches)
                    data["guest_team"] = readable_guest_matches
                    
                    # 计算战绩统计
                    guest_recent_10 = calculate_team_recent_10_games(sorted_guest_matches)
                    data["guest_team_recent10Games"] = guest_recent_10
                
                # 更新原始数据
                original_data["data"] = data
            
            return original_data
            
    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_match_recent_forms ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_match_recent_forms: [Type: {type(e).__name__}] - [Details: {repr(e)}]"

# http://ai-match.fengkuangtiyu.cn/api/v5/matches/3558764/squad_details
# 阵容（仅足球）
@mcp.tool()
async def get_football_squad_by_match_id(
        match_id: str
) -> Union[Dict[str, Any], str]:
    """
    根据比赛获取比赛阵容（仅限足球）。
    Fetches the squad details for a football match by its ID.

    Args:
        match_id (str): 比赛 (必需), 例如 '3558764'。

    Returns:
        Union[Dict[str, Any], str]: 成功时返回阵容数据，失败时返回错误信息字符串。
    """
    endpoint = f"http://ai-match.fengkuangtiyu.cn/api/v5/matches/{match_id}/squad_details"

    print(f"Executing get_squad_details for match_id: {match_id}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint)  # 这个接口不需要任何 query 参数
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_squad_details ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_squad_details: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


def calculate_h2h_summary(matches: List[Dict], team1_name: str, team2_name: str) -> Dict[str, Any]:
    """
    计算历史交锋汇总信息
    
    Args:
        matches: 历史交锋比赛列表
        team1_name: 队伍1名称
        team2_name: 队伍2名称
    
    Returns:
        包含total_matches, team1, team2, draw的汇总信息
    """
    if not matches or not isinstance(matches, list):
        return {
            "total_matches": 0,
            "team1": f"{team1_name} 0胜",
            "team2": f"{team2_name} 0胜", 
            "draw": 0
        }
    
    team1_wins = 0
    team2_wins = 0
    draws = 0
    
    for match in matches:
        if not isinstance(match, dict):
            continue
            
        result = match.get('result') or match.get('match_result')
        penalty_score = match.get('penalty_score') or match.get('penaltyScore', '')
        is_home = match.get('is_home') or match.get('isHome')
        home_team_name = match.get('home_team_name') or match.get('home_team', '')
        guest_team_name = match.get('guest_team_name') or match.get('away_team', '')
        
        # 确保result和is_home都是整数类型
        try:
            result = int(result) if result is not None else None
        except (ValueError, TypeError):
            result = None
            
        try:
            is_home = int(is_home) if is_home is not None else None
        except (ValueError, TypeError):
            is_home = None
            
        # 判断当前比赛中team1是主队还是客队
        team1_is_home = (home_team_name == team1_name)
        
        if result == 3:  # 胜
            if team1_is_home:
                team1_wins += 1
            else:
                team2_wins += 1
        elif result == 0:  # 负
            if team1_is_home:
                team2_wins += 1
            else:
                team1_wins += 1
        elif result == 1:  # 需要进一步判断
            if not penalty_score:  # 没有点球，则是平局
                draws += 1
            else:
                # 有点球，根据is_home和penalty_score判断胜负
                split_pattern = r'[\-–—−]'
                score_parts = re.split(split_pattern, penalty_score)
                if len(score_parts) == 2:
                    try:
                        left_score = int(score_parts[0].strip())
                        right_score = int(score_parts[1].strip())

                        # 主队点球胜
                        if is_home == 1 and left_score > right_score:
                            if team1_is_home:
                                team1_wins += 1
                            else:
                                team2_wins += 1
                        # 主队点球负
                        elif is_home == 1 and left_score < right_score:
                            if team1_is_home:
                                team2_wins += 1
                            else:
                                team1_wins += 1
                        # 客队点球胜
                        elif is_home == 0 and right_score > left_score:
                            if team1_is_home:
                                team2_wins += 1
                            else:
                                team1_wins += 1
                        # 客队点球负
                        elif is_home == 0 and right_score < left_score:
                            if team1_is_home:
                                team1_wins += 1
                            else:
                                team2_wins += 1
                        else:
                            draws += 1  # 默认计为平局

                    except (ValueError, AttributeError):
                        draws += 1  # 解析失败计为平局
                else:
                    draws += 1  # 格式异常计为平局
    
    total_matches = len(matches)
    
    return {
        "total_matches": total_matches,
        "team1": f"{team1_name} {team1_wins}胜",
        "team2": f"{team2_name} {team2_wins}胜",
        "draw": draws
    }


#  请调用 get_head_to_head_history_by_match_id 工具，参数为 match_id: "3558764"
@mcp.tool()
async def get_head_to_head_history_by_match_id(
        match_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        match_type: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """
        根据比赛获取双方历史交锋信息。默认查询近三个月的数据。
        Fetches the head-to-head (h2h) match history by a given match ID. Defaults to the last 3 months.

        Args:
            match_id (str): 比赛 (必需), 例如 '3558764'。
            start_date (Optional[str]): 筛选此日期之后的交锋记录，格式 YYYY-MM-DD。
            end_date (Optional[str]): 筛选此日期之前的交锋记录，格式 YYYY-MM-DD。
            match_type (Optional[str]): 体育类型, 1 代表足球, 2 代表篮球。

        Returns:
            Union[Dict[str, Any], str]: 成功时返回历史交锋数据，失败时返回错误信息字符串。
        """
    endpoint = f"http://ai-match.fengkuangtiyu.cn/api/v5/matches/{match_id}/h2h"

    # --- vvv 新增的默认日期处理 vvv ---
    # 如果调用者没有提供结束日期，则默认为今天
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # 如果调用者没有提供开始日期，则默认为三个月前的今天
    if not start_date:
        three_years_ago = datetime.now() - timedelta(days=3 * 30)
        start_date = three_years_ago.strftime('%Y-%m-%d')
    # --- ^^^ 新增的默认日期处理 ^^^ ---

    params = {"match_type": "1", "start_date": start_date, "end_date": end_date}
    if match_type:
        params["match_type"] = match_type

    # 注意：我们直接将处理后的 start_date 和 end_date 放入 params，不再需要 if 判断

    print(f"Executing get_head_to_head_history_by_match_id for match_id: {match_id} with params: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            original_data = response.json()
            
            # 如果API返回成功，添加历史交锋汇总信息
            if isinstance(original_data, dict) and original_data.get("code") == "0":
                data = original_data.get("data", {})
                
                # 获取比赛信息以确定队伍名称（兼容不同的字段名）
                matches = data.get("matches", []) or data.get("recent_matches", [])
                if matches and len(matches) > 0:
                    # 从第一场比赛获取队伍名称（兼容不同的字段名）
                    first_match = matches[0]
                    team1_name = first_match.get('home_team_name') or first_match.get('home_team', 'Team1')
                    team2_name = first_match.get('guest_team_name') or first_match.get('away_team', 'Team2')
                    
                    # 计算汇总信息
                    summary = calculate_h2h_summary(matches, team1_name, team2_name)
                    
                    # 将汇总信息添加到data层级
                    data["summary"] = summary
                else:
                    # 如果没有比赛数据，提供默认汇总
                    data["summary"] = {
                        "total_matches": 0,
                        "team1": "Team1 0胜",
                        "team2": "Team2 0胜",
                        "draw": 0
                    }
                
                # 更新原始数据
                original_data["data"] = data
            
            return original_data
            
    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_head_to_head_history_by_match_id ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_head_to_head_history_by_match_id: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


@mcp.tool()
async def get_europe_odds_by_match_id(
        match_id: str,
        match_type: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """
    根据比赛获取欧赔赔率信息（胜平负）。
    Fetches the European win-draw-lose odds for a match by its ID.

    Args:
        match_id (str): 比赛 (必需), 例如 '3558764'。
        match_type (Optional[str]): 体育类型, 1 代表足球, 2 代表篮球。

    Returns:
        Union[Dict[str, Any], str]: 成功时返回赔率数据，失败时返回错误信息字符串。
    """
    endpoint = f"http://ai-match.fengkuangtiyu.cn/api/v5/matches/{match_id}/odds/win_draw_lose"

    params = {"match_type": "1"}
    if match_type:
        params["match_type"] = match_type

    print(f"Executing get_win_draw_lose_odds for match_id: {match_id} with params: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_win_draw_lose_odds ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_europe_odds_by_match_id: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


@mcp.tool()
async def get_history_match(
        match_type: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """
    获取竞彩过去一周的比赛。
    Fetches competitive matches from the past week.

    Args:
        match_type (Optional[str]): 比赛类型, 1 代表竞彩足球, 2 代表竞彩篮球。

    Returns:
        Union[Dict[str, Any], str]: 成功时返回历史比赛数据，失败时返回错误信息字符串。
    """
    endpoint = "http://ai-match.fengkuangtiyu.cn/api/v5/matches/getHistoryMatch"

    params = {}
    if match_type:
        params["match_type"] = match_type

    print(f"Executing get_history_match with params: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_history_match ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_history_match: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


@mcp.tool()
async def get_asian_handicap_odds_by_match_id(
        match_id: str,
        match_type: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """
    根据比赛获取亚盘赔率信息。
    Fetches the Asian Handicap odds for a match by its ID.

    Args:
        match_id (str): 比赛 (必需), 例如 '3558764'。
        match_type (Optional[str]): 体育类型, 1 代表足球, 2 代表篮球。

    Returns:
        Union[Dict[str, Any], str]: 成功时返回赔率数据，失败时返回错误信息字符串。
    """
    endpoint = f"http://ai-match.fengkuangtiyu.cn/api/v5/matches/{match_id}/odds/asian_handicap"

    # --- 遵照你的要求，设置默认值 ---
    # 即使 AI 没有提供 match_type，我们也默认设置为 '1'
    params = {"match_type": "1"}
    # 如果 AI 提供了，就使用 AI 提供的值覆盖默认值
    if match_type:
        params["match_type"] = match_type
    # ---------------------------------

    print(f"Executing get_asian_handicap_odds for match_id: {match_id} with params: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_asian_handicap_odds ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_asian_handicap_odds: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


@mcp.tool()
async def get_over_under_odds_by_match_id(
        match_id: str,
        match_type: Optional[str] = None
) -> Union[Dict[str, Any], str]:
    """
    根据比赛获取足球大小球或篮球总分赔率。
    Fetches the Goal Line (Over/Under) or Total Points odds for a match by its ID.

    Args:
        match_id (str): 比赛 (必需), 例如 '3558764'。
        match_type (Optional[str]): 体育类型, 1 代表足球, 2 代表篮球。

    Returns:
        Union[Dict[str, Any], str]: 成功时返回赔率数据，失败时返回错误信息字符串。
    """
    endpoint = f"http://ai-match.fengkuangtiyu.cn/api/v5/matches/{match_id}/odds/goal_line"

    # 遵照你的要求，设置默认值
    params = {"match_type": "1"}
    if match_type:
        params["match_type"] = match_type

    print(f"Executing get_goal_line_odds for match_id: {match_id} with params: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_goal_line_odds ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_goal_line_odds: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


import asyncio  # 确保在文件顶部加上这个导入

# 辅助函数：清理冗余字段
def clean_redundant_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理数据中的冗余字段，如logo_url等
    """
    if isinstance(data, dict):
        # 移除常见的冗余字段
        redundant_fields = ['logo_url', 'team_logo', 'league_logo', 'avatar_url']
        cleaned_data = {}
        for key, value in data.items():
            if key not in redundant_fields:
                if isinstance(value, dict):
                    cleaned_data[key] = clean_redundant_fields(value)
                elif isinstance(value, list):
                    cleaned_data[key] = [clean_redundant_fields(item) if isinstance(item, dict) else item for item in value]
                else:
                    cleaned_data[key] = value
        return cleaned_data
    return data

def compress_odds_data(odds_data):
    """
    压缩赔率数据，将所有来源合并统计，显示最大值、最小值、平均值、中位数
    """
    if not odds_data:
        return {}
    
    import statistics
    
    def safe_float(value):
        """安全转换为浮点数，处理 None 值"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def calculate_stats(values):
        """计算统计数据"""
        # 过滤掉 None 值
        valid_values = [v for v in values if v is not None]
        if not valid_values:
            return {}
        try:
            return {
                'max': round(max(valid_values), 3),
                'min': round(min(valid_values), 3),
                'avg': round(statistics.mean(valid_values), 3),
                'median': round(statistics.median(valid_values), 3),
                'count': len(valid_values)
            }
        except:
            return {}
    
    compressed = {}
    
    # 压缩欧洲赔率 - 合并所有来源
    if 'europe' in odds_data and odds_data['europe']:
        europe_odds = odds_data['europe']
        
        # 收集所有来源的初始赔率
        home_initial = [safe_float(item['initial']['home']) for item in europe_odds if item.get('initial', {}).get('home')]
        draw_initial = [safe_float(item['initial']['draw']) for item in europe_odds if item.get('initial', {}).get('draw')]
        away_initial = [safe_float(item['initial']['away']) for item in europe_odds if item.get('initial', {}).get('away')]
        
        # 收集所有来源的即时赔率
        home_live = [safe_float(item['live']['home']) for item in europe_odds if item.get('live', {}).get('home')]
        draw_live = [safe_float(item['live']['draw']) for item in europe_odds if item.get('live', {}).get('draw')]
        away_live = [safe_float(item['live']['away']) for item in europe_odds if item.get('live', {}).get('away')]
        
        compressed['europe'] = {
            'initial': {
                'home': calculate_stats(home_initial),
                'draw': calculate_stats(draw_initial),
                'away': calculate_stats(away_initial)
            },
            'live': {
                'home': calculate_stats(home_live),
                'draw': calculate_stats(draw_live),
                'away': calculate_stats(away_live)
            }
        }
    
    # 压缩亚盘赔率 - 合并所有来源
    if 'asian_handicap' in odds_data and odds_data['asian_handicap']:
        asian_odds = odds_data['asian_handicap']
        
        # 收集所有来源的初始赔率
        line_initial = [safe_float(item['initial']['line']) for item in asian_odds if item.get('initial', {}).get('line')]
        home_initial = [safe_float(item['initial']['home_price']) for item in asian_odds if item.get('initial', {}).get('home_price')]
        away_initial = [safe_float(item['initial']['away_price']) for item in asian_odds if item.get('initial', {}).get('away_price')]
        
        # 收集所有来源的即时赔率
        line_live = [safe_float(item['live']['line']) for item in asian_odds if item.get('live', {}).get('line')]
        home_live = [safe_float(item['live']['home_price']) for item in asian_odds if item.get('live', {}).get('home_price')]
        away_live = [safe_float(item['live']['away_price']) for item in asian_odds if item.get('live', {}).get('away_price')]
        
        compressed['asian'] = {
            'initial': {
                'line': calculate_stats(line_initial),
                'home': calculate_stats(home_initial),
                'away': calculate_stats(away_initial)
            },
            'live': {
                'line': calculate_stats(line_live),
                'home': calculate_stats(home_live),
                'away': calculate_stats(away_live)
            }
        }
    
    # 压缩大小球赔率 - 合并所有来源
    if 'over_under' in odds_data and odds_data['over_under']:
        ou_odds = odds_data['over_under']
        
        # 收集所有来源的初始赔率
        line_initial = [safe_float(item['initial']['line']) for item in ou_odds if item.get('initial', {}).get('line')]
        over_initial = [safe_float(item['initial']['over_price']) for item in ou_odds if item.get('initial', {}).get('over_price')]
        under_initial = [safe_float(item['initial']['under_price']) for item in ou_odds if item.get('initial', {}).get('under_price')]
        
        # 收集所有来源的即时赔率
        line_live = [safe_float(item['live']['line']) for item in ou_odds if item.get('live', {}).get('line')]
        over_live = [safe_float(item['live']['over_price']) for item in ou_odds if item.get('live', {}).get('over_price')]
        under_live = [safe_float(item['live']['under_price']) for item in ou_odds if item.get('live', {}).get('under_price')]
        
        compressed['over_under'] = {
            'initial': {
                'line': calculate_stats(line_initial),
                'over': calculate_stats(over_initial),
                'under': calculate_stats(under_initial)
            },
            'live': {
                'line': calculate_stats(line_live),
                'over': calculate_stats(over_live),
                'under': calculate_stats(under_live)
            }
        }
    
    return compressed

def compress_recent_performance(performance_data):
    """
    压缩近期战绩数据，只显示统计结果，不显示比赛明细
    """
    if not performance_data:
        return {}
    
    compressed = {}
    
    # 压缩主队战绩 - 只保留统计结果
    if 'home_team' in performance_data:
        compressed['home'] = {
            'summary': performance_data.get('home_team_recent10Games', 'N/A')
        }
    
    # 压缩客队战绩 - 只保留统计结果
    if 'guest_team' in performance_data:
        compressed['away'] = {
            'summary': performance_data.get('guest_team_recent10Games', 'N/A')
        }
    
    return compressed

def compress_squad_data(squad_data):
    """
    压缩阵容数据，只显示统计信息，不显示队员明细
    """
    if not squad_data:
        return {}
    
    def analyze_team_squad(team_data):
        """分析单个队伍的阵容统计"""
        if not team_data:
            return {
                "total_players": 0,
                "key_players": 0,
                "injuries": 0,
                "position_distribution": {},
                "formation": "N/A"
            }
        
        # 统计位置分布
        position_count = {}
        total_players = 0
        
        # 统计关键球员
        key_players = team_data.get('key_player', [])
        last_key_players = team_data.get('last_key_player', [])
        
        # 合并所有球员进行位置统计
        all_players = key_players + last_key_players
        for player in all_players:
            position = player.get('position', '未知')
            position_count[position] = position_count.get(position, 0) + 1
            total_players += 1
        
        # 去重统计（避免重复计算同一球员）
        unique_players = set()
        for player in all_players:
            unique_players.add(player.get('player_name', ''))
        
        return {
            "total_players": len(unique_players),
            "key_players": len(key_players),
            "injuries": len(team_data.get('injury', [])),
            "position_distribution": position_count,
            "formation": team_data.get('formation', 'N/A')
        }
    
    compressed = {}
    
    # 压缩主队数据
    if 'home_team' in squad_data:
        compressed['home'] = analyze_team_squad(squad_data['home_team'])
    
    # 压缩客队数据  
    if 'away_team' in squad_data:
        compressed['away'] = analyze_team_squad(squad_data['away_team'])
    
    return compressed

def compress_h2h_data(h2h_data):
    """
    压缩历史交锋数据，格式：[日期, 主队, 客队, 比分, 结果]
    """
    if not h2h_data or 'matches' not in h2h_data:
        return {}
    
    matches = h2h_data['matches']
    compressed = {
        'summary': h2h_data.get('summary', {}),
        'matches': [[m['date'], m['home_team'], m['away_team'], m['score'], m['result']] for m in matches[:5]]  # 只保留最近5场
    }
    
    return compressed

async def _get_single_match_info(
        match_id: str,
        match_type: Optional[str] = None,
) -> Union[Dict[str, Any], str]:
    """
    获取比赛的所有综合信息，包括基本信息、排名、近期战绩、历史交锋、阵容和赔率等。
    Fetches comprehensive match information including details, standings, recent performance, head-to-head history, squad, and odds.

    Args:
        match_id (str): 比赛ID (必需), 例如 '3642247'。
        match_type (Optional[str]): 体育类型, 1 代表足球, 2 代表篮球。默认为1。

    Returns:
        Union[Dict[str, Any], str]: 成功时返回综合比赛信息，失败时返回错误信息字符串。
    """
    # 设置默认match_type
    if not match_type:
        match_type = "1"
    compress = True
    print(f"Executing get_comprehensive_match_info for match_id: {match_id} with match_type: {match_type}")
    
    # 存储所有数据
    comprehensive_data = {
        "match_id": match_id,
        "match_type": match_type,
        "basic_info": None,
        "standings": None,
        "recent_performance": None,
        "head_to_head": None,
        "squad": None,
        "odds": {
            "europe": None,
            "asian_handicap": None,
            "over_under": None
        },
        "errors": []
    }
    
    try:
        # 1. 获取比赛基本信息
        try:
            basic_info = await get_match_details_by_id(match_id, match_type)
            if isinstance(basic_info, dict) and basic_info.get("code") == "0":
                comprehensive_data["basic_info"] = clean_redundant_fields(basic_info.get("data", {}))
            else:
                comprehensive_data["errors"].append(f"基本信息获取失败: {basic_info}")
        except Exception as e:
            comprehensive_data["errors"].append(f"基本信息获取异常: {str(e)}")
        
        # 2. 获取联赛排名
        try:
            standings = await get_match_standings_by_id(match_id, match_type)
            if isinstance(standings, dict) and standings.get("code") == "0":
                comprehensive_data["standings"] = clean_redundant_fields(standings.get("data", {}))
            else:
                comprehensive_data["errors"].append(f"排名信息获取失败: {standings}")
        except Exception as e:
            comprehensive_data["errors"].append(f"排名信息获取异常: {str(e)}")
        
        # 3. 获取近期战绩
        try:
            recent_performance = await get_team_recent_performance_by_match_id(match_id, match_type)
            if isinstance(recent_performance, dict) and recent_performance.get("code") == "0":
                comprehensive_data["recent_performance"] = clean_redundant_fields(recent_performance.get("data", {}))
            else:
                comprehensive_data["errors"].append(f"近期战绩获取失败: {recent_performance}")
        except Exception as e:
            comprehensive_data["errors"].append(f"近期战绩获取异常: {str(e)}")
        
        # 4. 获取历史交锋
        try:
            head_to_head = await get_head_to_head_history_by_match_id(match_id, None, None, match_type)
            if isinstance(head_to_head, dict) and head_to_head.get("code") == "0":
                comprehensive_data["head_to_head"] = clean_redundant_fields(head_to_head.get("data", {}))
            else:
                comprehensive_data["errors"].append(f"历史交锋获取失败: {head_to_head}")
        except Exception as e:
            comprehensive_data["errors"].append(f"历史交锋获取异常: {str(e)}")
        
        # 5. 获取阵容信息（仅足球）
        if match_type == "1":
            try:
                squad = await get_football_squad_by_match_id(match_id)
                if isinstance(squad, dict) and squad.get("code") == "0":
                    comprehensive_data["squad"] = clean_redundant_fields(squad.get("data", {}))
                else:
                    comprehensive_data["errors"].append(f"阵容信息获取失败: {squad}")
            except Exception as e:
                comprehensive_data["errors"].append(f"阵容信息获取异常: {str(e)}")
        
        # 6. 获取欧洲赔率
        try:
            europe_odds = await get_europe_odds_by_match_id(match_id, match_type)
            if isinstance(europe_odds, dict) and europe_odds.get("code") == "0":
                comprehensive_data["odds"]["europe"] = clean_redundant_fields(europe_odds.get("data", {}))
            else:
                comprehensive_data["errors"].append(f"欧洲赔率获取失败: {europe_odds}")
        except Exception as e:
            comprehensive_data["errors"].append(f"欧洲赔率获取异常: {str(e)}")
        
        # 7. 获取亚盘赔率
        try:
            asian_odds = await get_asian_handicap_odds_by_match_id(match_id, match_type)
            if isinstance(asian_odds, dict) and asian_odds.get("code") == "0":
                comprehensive_data["odds"]["asian_handicap"] = clean_redundant_fields(asian_odds.get("data", {}))
            else:
                comprehensive_data["errors"].append(f"亚盘赔率获取失败: {asian_odds}")
        except Exception as e:
            comprehensive_data["errors"].append(f"亚盘赔率获取异常: {str(e)}")
        
        # 8. 获取大小球赔率
        try:
            over_under_odds = await get_over_under_odds_by_match_id(match_id, match_type)
            if isinstance(over_under_odds, dict) and over_under_odds.get("code") == "0":
                comprehensive_data["odds"]["over_under"] = clean_redundant_fields(over_under_odds.get("data", {}))
            else:
                comprehensive_data["errors"].append(f"大小球赔率获取失败: {over_under_odds}")
        except Exception as e:
            comprehensive_data["errors"].append(f"大小球赔率获取异常: {str(e)}")
        
        # 统计成功获取的数据项
        successful_items = []
        if comprehensive_data["basic_info"]: successful_items.append("基本信息")
        if comprehensive_data["standings"]: successful_items.append("排名信息")
        if comprehensive_data["recent_performance"]: successful_items.append("近期战绩")
        if comprehensive_data["head_to_head"]: successful_items.append("历史交锋")
        if comprehensive_data["squad"]: successful_items.append("阵容信息")
        if comprehensive_data["odds"]["europe"]: successful_items.append("欧洲赔率")
        if comprehensive_data["odds"]["asian_handicap"]: successful_items.append("亚盘赔率")
        if comprehensive_data["odds"]["over_under"]: successful_items.append("大小球赔率")
        
        # 提取关键信息到顶层，便于快速访问
        key_info = {}
        if comprehensive_data["basic_info"]:
            basic = comprehensive_data["basic_info"]
            key_info.update({
                "home_team": basic.get('home_team_name', 'N/A'),
                "away_team": basic.get('away_team_name', 'N/A'),
                "match_time": basic.get('match_time_utc', 'N/A'),
                "league_name": basic.get('league_name', 'N/A'),
                "season": basic.get('season', 'N/A'),
                "match_status": basic.get('status', 'N/A')
            })
        
        # 添加近期战绩汇总
        if comprehensive_data["recent_performance"]:
            recent = comprehensive_data["recent_performance"]
            key_info.update({
                "home_team_form": recent.get('home_team_recent10Games', 'N/A'),
                "away_team_form": recent.get('guest_team_recent10Games', 'N/A')
            })
        
        # 添加历史交锋汇总
        if comprehensive_data["head_to_head"]:
            h2h = comprehensive_data["head_to_head"]
            h2h_summary = h2h.get('summary', {})
            key_info.update({
                "h2h_total_matches": h2h_summary.get('total_matches', 0),
                "h2h_summary": f"{h2h_summary.get('team1', 'N/A')} vs {h2h_summary.get('team2', 'N/A')}"
            })
        
        comprehensive_data["key_info"] = key_info
        
        print(f"综合信息获取完成 - 成功: {len(successful_items)}项, 错误: {len(comprehensive_data['errors'])}项")
        
        # 如果启用压缩，则压缩数据
        if compress:
            print("启用数据压缩模式...")
            
            # 压缩赔率数据
            if comprehensive_data["odds"]:
                comprehensive_data["odds"] = compress_odds_data(comprehensive_data["odds"])
            
            # 压缩近期战绩数据
            if comprehensive_data["recent_performance"]:
                comprehensive_data["recent_performance"] = compress_recent_performance(comprehensive_data["recent_performance"])
            
            # 压缩阵容数据
            if comprehensive_data["squad"]:
                comprehensive_data["squad"] = compress_squad_data(comprehensive_data["squad"])
            
            # 压缩历史交锋数据
            if comprehensive_data["head_to_head"]:
                comprehensive_data["head_to_head"] = compress_h2h_data(comprehensive_data["head_to_head"])
            
            print("数据压缩完成")
        
        return comprehensive_data
        
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_comprehensive_match_info ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_comprehensive_match_info: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


@mcp.tool()
async def get_comprehensive_matches_info(
        match_ids: List[str],
        match_type: Optional[str] = None,
) -> Union[Dict[str, Any], str]:
    """
    获取比赛的综合信息，支持单个或多个比赛ID，自动启用并发处理提高性能。
    Fetches comprehensive match information for single or multiple matches with automatic concurrent processing.

    Args:
        match_ids (List[str]): 比赛ID列表，必须是字符串列表 (必需)
                              单个比赛ID也需要传递长度为1的列表
                              例如: ['3642247'] 或 ['3642247', '3642248', '3642249']
        match_type (Optional[str]): 体育类型, 1 代表足球, 2 代表篮球。默认为1。

    Returns:
        Union[Dict[str, Any], str]: 单个ID时返回比赛信息字典，多个ID时返回批量结果字典，失败时返回错误信息字符串。
    """
    # 设置默认match_type
    if not match_type:
        match_type = "1"
    
    try:
        # 处理输入参数，确保是字符串列表
        if not isinstance(match_ids, list):
            return "错误: match_ids 必须是字符串列表"
        
        id_list = [str(mid).strip() for mid in match_ids if str(mid).strip()]
        is_single_request = len(id_list) == 1
        
        if not id_list:
            return "错误: match_ids 不能为空"
        
        print(f"Executing get_comprehensive_match_info for {len(id_list)} matches with match_type: {match_type}")
        
        # 单个比赛直接处理
        if len(id_list) == 1:
            result = await _get_single_match_info(id_list[0], match_type)
            return result
        
        # 多个比赛使用并发处理
        print(f"使用并发模式处理 {len(id_list)} 个比赛...")
        
        # 批量处理结果
        batch_results = {
            "total_matches": len(id_list),
            "successful_matches": 0,
            "failed_matches": 0,
            "results": [],
            "processing_summary": {
                "concurrent_enabled": True,
                "match_type": match_type,
                "processed_ids": id_list
            }
        }
        
        # 创建并发任务
        tasks = []
        for match_id in id_list:
            task = _get_single_match_info(match_id, match_type)
            tasks.append(task)
        
        # 执行并发任务
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for i, result in enumerate(results):
                match_id = id_list[i]
                if isinstance(result, Exception):
                    print(f"比赛 {match_id} 处理失败: {str(result)}")
                    batch_results["results"].append({
                        "match_id": match_id,
                        "success": False,
                        "error": str(result),
                        "data": None
                    })
                    batch_results["failed_matches"] += 1
                elif isinstance(result, str):
                    # 字符串表示错误
                    print(f"比赛 {match_id} 处理失败: {result}")
                    batch_results["results"].append({
                        "match_id": match_id,
                        "success": False,
                        "error": result,
                        "data": None
                    })
                    batch_results["failed_matches"] += 1
                else:
                    # 成功获取数据
                    print(f"比赛 {match_id} 处理成功")
                    batch_results["results"].append({
                        "match_id": match_id,
                        "success": True,
                        "error": None,
                        "data": result
                    })
                    batch_results["successful_matches"] += 1
                    
        except Exception as e:
            import traceback
            print(f"--- DETAILED ERROR IN concurrent processing ---")
            traceback.print_exc()
            return f"并发处理失败: [Type: {type(e).__name__}] - [Details: {repr(e)}]"
        
        # 添加处理汇总信息
        batch_results["processing_summary"].update({
            "success_rate": f"{batch_results['successful_matches']}/{batch_results['total_matches']} ({batch_results['successful_matches']/batch_results['total_matches']*100:.1f}%)",
            "completion_status": "completed"
        })
        
        print(f"批量处理完成 - 成功: {batch_results['successful_matches']}, 失败: {batch_results['failed_matches']}")
        
        return batch_results
        
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_comprehensive_match_info ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_comprehensive_match_info: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


# 生成比赛统计信息的辅助函数"""
# 你是一名专业的足球比赛数据分析师。请对比赛 ID 为 3551089 的比赛进行一次全面的赛前分析。请利用你可用的工具获取所有必要信息，并基于这些信息，为我撰写一份包含基本面分析、近期战绩和交锋分析以及最终结论和比分预测的专业报告。
# """
# @mcp.tool()
# async def generate_full_match_report_by_id(match_id: str) -> Union[Dict[str, Any], str]:
#     """
#     根据比赛，生成一份包含所有可用信息的完整比赛分析报告。这是进行比赛分析的首选工具。
#     它会自动处理ID有效性检查，如果ID无效，会直接返回错误信息。
#     Generates a full pre-match analysis report with all available information for a given match ID. This is the preferred tool for match analysis.
#     It automatically handles the validity check of the ID and returns an error message if the ID is invalid.

#     Args:
#         match_id (str): 比赛 (必需), 例如 '3558764'。

#     Returns:
#         Union[Dict[str, Any], str]: 一份包含多维度数据的完整报告，或明确的错误信息字符串。
#     """
#     print(f"🚀 [START] Generating full report for match_id: {match_id}")

#     try:
#         # --- 优化点 1: 优先进行核心数据检查 ---
#         # 首先，只调用最核心的 get_match_details_by_id。
#         # 这一步既能获取基础信息，又能充当“有效性检查”。
#         # FIX: Switched to triple quotes for multi-line f-string.
#         print(f"""step 1:
#   - Checking match validity and fetching essential details...""")
#         match_details = await get_match_details_by_id(match_id=match_id)

#         # 如果返回的是字符串，说明API调用失败（例如404 Not Found），ID很可能无效。
#         if isinstance(match_details, str):
#             print(f"❌ [STOP] Invalid match_id or API error. Reason: {match_details}")
#             # 直接将这个清晰的错误返回给 AI，AI 就能理解比赛有问题。
#             return f"无法为比赛 '{match_id}' 生成报告，因为它可能是无效的或数据暂时不可用。错误详情: {match_details}"

#         # 如果代码能走到这里，说明 ID 是有效的，match_details 是一个包含数据的 dict。
#         # FIX: Switched to triple quotes for multi-line f-string.
#         print(f"""✅ step 1:
#   - Match ID is valid. Home: {match_details.get('data', {}).get('home_team_name')}, Away: {match_details.get('data', {}).get('away_team_name')}""")

#         # --- 优化点 2: 并发获取剩余数据 ---
#         # 现在我们确认ID有效，可以安全地并发获取所有其他辅助数据。
#         # FIX: Switched to triple quotes for multi-line f-string.
#         print(f"""step 2:
#   - Concurrently fetching all other data points...""")
#         other_data_tasks = [
#             get_match_standings_by_id(match_id=match_id),
#             get_team_recent_performance_by_match_id(match_id=match_id),
#             get_head_to_head_history_by_match_id(match_id=match_id),
#             get_football_squad_by_match_id(match_id=match_id),
#             get_europe_odds_by_match_id(match_id=match_id),
#             get_asian_handicap_odds_by_match_id(match_id=match_id),
#             get_over_under_odds_by_match_id(match_id=match_id),
#         ]

#         # 使用 asyncio.gather 并发执行，并设置 return_exceptions=True
#         # 这样即使某个API（比如阵容）失败，也不会中断整个流程。
#         results = await asyncio.gather(*other_data_tasks, return_exceptions=True)
#         # FIX: Switched to triple quotes for multi-line f-string.
#         print(f"""✅ step 2:
#   - All data fetching tasks completed.""")

#         # --- 优化点 3: 结构化地组装报告 ---
#         # 将所有成功或失败的结果清晰地组装起来。
#         report = {
#             # 核心数据已在第一步获取
#             "match_details": match_details,
#             # 其他数据
#             "standings": results[0],
#             "recent_performance": results[1],
#             "h2h_history": results[2],
#             "squad": results[3],
#             "europe_odds": results[4],
#             "asian_handicap_odds": results[5],
#             "over_under_odds": results[6],
#         }

#         # 检查是否有失败的调用，并将异常转换为可读的错误信息
#         for key, value in report.items():
#             if isinstance(value, Exception):
#                 error_message = f"Failed to fetch {key}: {str(value)}"
#                 report[key] = {"error": error_message}  # 使用结构化的错误
#                 print(f"⚠️ Warning: Partial data failure for '{key}'. Reason: {error_message}")

#         print(f"✅ [SUCCESS] Full report generated for match_id: {match_id}")
#         return report

#     except Exception as e:
#         # 捕获意外的顶层异常
#         error_details = f"An unexpected error occurred in generate_full_match_report_by_id: [Type: {type(e).__name__}] - [Details: {repr(e)}]"
#         traceback.print_exc()
#         print(f"❌ [FATAL] {error_details}")
#         return f"生成完整报告时发生严重错误: {error_details}"


# 请调用 get_upcoming_competitive_matches 工具，match_type 为 "1"，直接调用，不要使用其他工具
def _generate_match_statistics(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    生成比赛统计信息
    
    Args:
        matches: 比赛列表
        
    Returns:
        Dict: 包含统计信息的字典
    """
    from collections import Counter
    from datetime import datetime, timedelta
    
    if not matches:
        return {
            "total_matches": 0,
            "league_distribution": {},
            "time_distribution": {},
            "upcoming_days": 0
        }
    
    # 统计联赛分布
    league_counter = Counter()
    time_counter = Counter()
    match_dates = []
    
    for match in matches:
        # 联赛统计
        league_name = match.get('league_name', '未知联赛')
        league_counter[league_name] += 1
        
        # 时间统计
        match_date = match.get('date')
        if match_date:
            try:
                # 解析时间
                if isinstance(match_date, str):
                    # 处理不同的时间格式
                    if 'T' in match_date:
                        dt = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                    else:
                        dt = datetime.strptime(match_date, '%Y-%m-%d %H:%M:%S')
                else:
                    dt = match_date
                
                match_dates.append(dt)
                
                # 按日期分组
                date_str = dt.strftime('%Y-%m-%d')
                time_counter[date_str] += 1
                
            except Exception as e:
                print(f"Error parsing date {match_date}: {e}")
    
    # 计算未来天数范围
    upcoming_days = 0
    if match_dates:
        now = datetime.now()
        # 移除时区信息进行比较
        match_dates_naive = []
        for dt in match_dates:
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            match_dates_naive.append(dt)
        
        if match_dates_naive:
            latest_date = max(match_dates_naive)
            upcoming_days = (latest_date - now).days + 1
            upcoming_days = max(0, upcoming_days)  # 确保不为负数
    
    return {
        "total_matches": len(matches),
        "league_distribution": dict(league_counter),
        "time_distribution": dict(time_counter),
        "upcoming_days": upcoming_days
    }


@mcp.tool()
async def get_upcoming_competitive_matches(
        match_type: str
) -> Union[Dict[str, Any], str]:
    """
    获取竞彩足球和篮球未开始的比赛列表，包含所有比赛的统计信息和最临近的10场比赛详情。
    Fetches upcoming competitive football and basketball matches with comprehensive statistics and details of the 10 closest matches.

    Args:
        match_type (str): 体育类型，必需参数。1 代表竞彩足球, 2 代表竞彩篮球。

    Returns:
        Union[Dict[str, Any], str]: 成功时返回包含统计信息和比赛列表的字典，失败时返回错误信息字符串。
        返回结构：
        {
            "statistics": {
                "total_matches": int,
                "league_distribution": Dict[str, int],
                "time_distribution": Dict[str, int],
                "upcoming_days": int
            },
            "upcoming_matches": List[Dict[str, Any]]
        }
    """
    endpoint = "http://ai-match.fengkuangtiyu.cn/api/v5/matches/getNotStartMatch"
    params = {"match_type": match_type}
    
    print(f"Executing get_upcoming_competitive_matches with params: {params}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()

            data = response.json()
            # Check if the API call was successful
            if data.get("code") == "0":
                matches = data.get("data", [])
                
                # 如果没有比赛数据，返回空结构
                if not matches:
                    return {
                        "statistics": {
                            "total_matches": 0,
                            "league_distribution": {},
                            "time_distribution": {},
                            "upcoming_days": 0
                        },
                        "upcoming_matches": []
                    }
                
                # 为每场比赛并发获取详细信息
                print(f"开始并发获取 {len(matches)} 场比赛的详细信息...")
                
                # 创建并发任务
                async def get_single_match_detail(match):
                    match_id = match.get('ID')
                    if not match_id:
                        return None
                    
                    try:
                        match_details = await get_match_details_by_id(str(match_id), match_type)
                        if isinstance(match_details, dict) and match_details.get('code') == '0':
                            return match, match_details
                        else:
                            print(f"获取比赛 {match_id} 详情失败: {match_details}")
                            return None
                    except Exception as e:
                        print(f"获取比赛 {match_id} 详情时发生错误: {e}")
                        traceback.print_exc()
                        return None
                
                # 并发执行所有任务
                tasks = [get_single_match_detail(match) for match in matches]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果
                detailed_matches = []
                successful_count = 0
                failed_count = 0
                
                for result in results:
                    if isinstance(result, Exception):
                        print(f"并发任务执行异常: {result}")
                        traceback.print_exc()
                        failed_count += 1
                        continue
                    
                    if result is None:
                        failed_count += 1
                        continue
                    
                    match, match_details = result
                    successful_count += 1
                    
                    # 提取详细信息中的 data 字段
                    detail_data = match_details.get('data', {})
                    
                    # 合并基本信息和详细信息，去除重复字段
                    combined_match = {}
                    
                    # 先添加原始数据
                    combined_match.update(match)
                    
                    # 然后添加详细信息，覆盖重复字段
                    if detail_data:
                        # 添加联赛名称
                        if 'league_name' in detail_data:
                            combined_match['league_name'] = detail_data['league_name']
                        
                        # 添加比赛时间
                        if 'match_time_utc' in detail_data:
                            combined_match['date'] = detail_data['match_time_utc']
                        
                        # 添加状态
                        if 'status' in detail_data:
                            combined_match['status'] = detail_data['status']
                        
                        # 添加赛季信息
                        if 'season' in detail_data:
                            combined_match['season'] = detail_data['season']
                        
                        # 添加队伍详细信息
                        if 'teams' in detail_data:
                            teams = detail_data['teams']
                            if 'home' in teams and 'name' in teams['home']:
                                combined_match['HOST_NAME'] = teams['home']['name']
                            if 'away' in teams and 'name' in teams['away']:
                                combined_match['GUEST_NAME'] = teams['away']['name']
                    
                    detailed_matches.append(combined_match)
                
                print(f"并发获取详情完成: 成功 {successful_count} 场，失败 {failed_count} 场")
                
                # 生成统计信息
                statistics = _generate_match_statistics(detailed_matches)
                
                # 按时间排序并限制为最临近的10场比赛
                sorted_matches = sort_and_limit_matches(detailed_matches, limit=10, upcoming=True)
                
                # 标准化字段名称，添加常用的字段映射
                for match in sorted_matches:
                    # 添加标准化的队伍名称字段
                    if 'GUEST_NAME' in match:
                        match['away_team_name'] = match['GUEST_NAME']
                    if 'HOST_NAME' in match:
                        match['home_team_name'] = match['HOST_NAME']
                    if 'ID' in match:
                        match['match_id'] = match['ID']
                
                return {
                    "statistics": statistics,
                    "upcoming_matches": sorted_matches
                }
            else:
                # If the API returned an error, return error message
                return f"API returned an error: {data.get('msg', 'Unknown error')}"

    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        print(f"--- DETAILED ERROR IN get_upcoming_competitive_matches ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_upcoming_competitive_matches: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


# 获取历史任九期次
# http://ai-match.fengkuangtiyu.cn/api/v5/matches/getHistoryNineMatchIssueList
@mcp.tool()
async def get_history_nine_match_issue_list() -> Union[Dict[str, Any], str]:
    """
    获取历史任九期次列表。
    Fetches the list of historical nine-match lottery issues.

    Returns:
        Union[Dict[str, Any], str]: 成功时返回历史期次数据，失败时返回错误信息字符串。
    """
    endpoint = "http://ai-match.fengkuangtiyu.cn/api/v5/matches/getHistoryNineMatchIssueList"

    print(f"Executing get_history_nine_match_issue_list")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_history_nine_match_issue_list ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_history_nine_match_issue_list: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


# 获取未结束任九期次
# http://ai-match.fengkuangtiyu.cn/api/v5/matches/getNotStartNineMatchIssueList
@mcp.tool()
async def get_not_start_nine_match_issue_list() -> Union[Dict[str, Any], str]:
    """
    获取未结束任九期次列表。
    Fetches the list of unfinished nine-match lottery issues.

    Returns:
        Union[Dict[str, Any], str]: 成功时返回未结束期次数据，失败时返回错误信息字符串。
    """
    endpoint = "http://ai-match.fengkuangtiyu.cn/api/v5/matches/getNotStartNineMatchIssueList"

    print(f"Executing get_not_start_nine_match_issue_list")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_not_start_nine_match_issue_list ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_not_start_nine_match_issue_list: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


# 获取任九期次比赛列表
# http://ai-match.fengkuangtiyu.cn/api/v5/matches/getIssueMatchList?issue=2025132
@mcp.tool()
async def get_issue_match_list(issue: str) -> Union[Dict[str, Any], str]:
    """
    获取任九期次比赛列表。
    Fetches the match list for a specific nine-match lottery issue.

    Args:
        issue (str): 期次（必填），例如 '2025132'。

    Returns:
        Union[Dict[str, Any], str]: 成功时返回期次比赛数据，失败时返回错误信息字符串。
    """
    endpoint = "http://ai-match.fengkuangtiyu.cn/api/v5/matches/getIssueMatchList"
    params = {"issue": issue}

    print(f"Executing get_issue_match_list with params: {params}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return f"API request failed with status {e.response.status_code}: {e.response.text}"
    except Exception as e:
        import traceback
        print(f"--- DETAILED ERROR IN get_issue_match_list ---")
        traceback.print_exc()
        return f"An unexpected error occurred in get_issue_match_list: [Type: {type(e).__name__}] - [Details: {repr(e)}]"


# 4. 启动 Web 服务器的代码
def create_app():
    """创建Starlette应用"""
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    
    async def health_check(request):
        """健康检查端点"""
        return JSONResponse({"status": "ok", "service": "Match Data MCP Server"})
    
    # 创建应用，包含健康检查和MCP SSE端点
    app = Starlette(
        routes=[
            Route('/health', health_check),
            Mount('/', app=mcp.sse_app()),
        ]
    )
    
    return app

if __name__ == "__main__":
    port = 34012  # 改为34012避免端口冲突
    print("🏆 Starting Match Data MCP Server...")
    print(f"📡 Server will be available at: http://0.0.0.0:{port}")
    print(f"🌐 SSE endpoint: http://0.0.0.0:{port}/sse")
    print(f"❤️ Health check: http://0.0.0.0:{port}/health")
    
    app = create_app()
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        traceback.print_exc()