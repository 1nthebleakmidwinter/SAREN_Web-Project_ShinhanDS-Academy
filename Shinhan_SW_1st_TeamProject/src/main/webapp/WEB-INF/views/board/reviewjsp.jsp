<%@ page language="java" contentType="text/html; charset=EUC-KR"
    pageEncoding="EUC-KR"%>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Header</title>
    <!-- css -->
    <link rel="stylesheet" href="css/style.css">
    <link rel="stylesheet" href="css/header_footer.css">
    <link rel="stylesheet" href="css/common.css">
    <style>
        body {
            font-family: Arial, sans-serif;
        }

        .inner {
            width: 80%;
            margin: 0 auto;
        }

        .review-container {
            margin-top: 40px;
            display: flex;
        }

        .sidebar {
            width: 150px;
            margin-right: 20px;
        }

        .sidebar ul {
            list-style: none;
            padding: 0;
        }

        .sidebar ul li {
            margin: 10px 0;
        }

        .sidebar ul li a {
            text-decoration: none;
            color: #000;
            font-size: 16px;
        }

        .review-content {
            flex-grow: 1;
            margin-left: 20px;
        }

        .review-header {
            text-align: center;
            border-bottom: 2px solid #000;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }

        .review-header h2 {
            margin: 0;
            font-size: 20px;
            font-weight: bold;
        }

        .review-form {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .review-form label {
            width: 100%;
            margin-bottom: 10px;
        }

        .review-form label span {
            display: inline-block;
            width: 100px;
            font-weight: bold;
        }

        .review-form input[type="text"],
        .review-form textarea {
            width: calc(100% - 110px);
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
        }

        .review-form textarea {
            height: 100px;
        }

        .review-form .rating {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }

        .review-form .rating span {
            font-weight: bold;
            margin-right: 10px;
        }

        .review-form .rating input[type="radio"] {
            margin-right: 5px;
        }

        .review-form img {
            width: 200px;
            height: auto;
            margin-bottom: 10px;
        }

        .review-form button {
            background-color: #6a0dad;
            color: #fff;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
        }

        .review-form button:hover {
            background-color: #5a0cac;
        }
    </style>
</head>

<body>
    <!-- header -->
    <header>
        <div class="header_top inner">
            <div class="leftGnb">
                <h1 class="logo">
                    <img src="images/logo_black.png" alt="�ΰ�">
                </h1>
                <form class="search_form">
                    <label for="search_wrap"> <input name="search_input" type="search" /> <img src="images/icon_serch.png" alt="������ ������" class="search_img" />
                    </label>
                </form>
            </div>
            <div class="rightGnb">
                <ul>
                    <li><a href="#"> <img src="images/icon-login.gif" alt="�α���">�α���
                    </a></li>
                    <li><a href="#">ȸ������</a></li>
                    <li><a href="#">����������</a></li>
                    <li><a href="#">���ι�</a></li>
                </ul>
            </div>
        </div>
        <div class="header_bottom">
            <div class="menu_wrap inner">
                <ul class="left_menu">
                    <li><a href="javascript:#void">����</a>
                        <div class="dropdown_nav">
                            <ul>
                                <li><a href="#">�Ż�ǰ</a></li>
                                <li><a href="#">��ü ��ǰ</a></li>
                                <li><a href="#">�ƿ���</a></li>
                                <li><a href="#">��Ŷ/����Ʈ</a></li>
                                <li><a href="#">��Ʈ</a></li>
                                <li><a href="#">����/���콺</a></li>
                                <li><a href="#">Ƽ����</a></li>
                                <li><a href="#">���ǽ�</a></li>
                                <li><a href="#">����</a></li>
                                <li><a href="#">��ĿƮ</a></li>
                                <li><a href="#">���/�ð�</a></li>
                            </ul>
                        </div>
                    </li>
                    <li><a href="javascript:#void">����</a>
                        <div class="dropdown_nav">
                            <ul>
                                <li><a href="#">�Ż�ǰ</a></li>
                                <li><a href="#">��ü ��ǰ</a></li>
                                <li><a href="#">�ƿ���</a></li>
                                <li><a href="#">��Ŷ/����Ʈ</a></li>
                                <li><a href="#">��Ʈ</a></li>
                                <li><a href="#">����/���콺</a></li>
                                <li><a href="#">Ƽ����</a></li>
                                <li><a href="#">���ǽ�</a></li>
                                <li><a href="#">����</a></li>
                                <li><a href="#">��ĿƮ</a></li>
                                <li><a href="#">���/�ð�</a></li>
                            </ul>
                        </div>
                    </li>
                    <li><a href="javascript:#void">Ű��</a>
                        <div class="dropdown_nav">
                            <ul>
                                <li><a href="#">�Ż�ǰ</a></li>
                                <li><a href="#">��ü ��ǰ</a></li>
                                <li><a href="#">����</a></li>
                                <li><a href="#">����</a></li>
                            </ul>
                        </div>
                    </li>
                    <li><a href="javascript:#void">���Ÿ�</a>
                        <div class="dropdown_nav">
                            <ul>
                                <li><a href="#">�Ż�ǰ</a></li>
                                <li><a href="#">��ü ��ǰ</a></li>
                                <li><a href="#">�����Ƿ�</a></li>
                                <li><a href="#">��������/����</a></li>
                                <li><a href="#">��������</a></li>
                                <li><a href="#">���� ���/�ð�</a></li>
                                <li><a href="#">�����Ƿ�</a></li>
                                <li><a href="#">��������/����</a></li>
                                <li><a href="#">��������</a></li>
                                <li><a href="#">���۶�/�Ȱ���</a></li>
                            </ul>
                        </div>
                    </li>
                    <li><a href="javascript:#void">Ű��</a>
                        <div class="dropdown_nav">
                            <ul>
                                <li><a href="#">�Ż�ǰ</a></li>
                                <li><a href="#">��ü ��ǰ</a></li>
                                <li><a href="#">�ƿ���</a></li>
                                <li><a href="#">��Ŷ/����Ʈ</a></li>
                                <li><a href="#">��Ʈ</a></li>
                                <li><a href="#">����/���콺</a></li>
                                <li><a href="#">Ƽ����</a></li>
                                <li><a href="#">���ǽ�</a></li>
                                <li><a href="#">����</a></li>
                                <li><a href="#">��ĿƮ</a></li>
                                <li><a href="#">���/�ð�</a></li>
                            </ul>
                        </div>
                    </li>
                    <li><a href="javascript:#void">������</a>
                        <div class="dropdown_nav">
                            <ul>
                                <li><a href="#">�Ż�ǰ</a></li>
                                <li><a href="#">��ü ��ǰ</a></li>
                                <li><a href="#">�ƿ�����/ķ��</a></li>
                                <li><a href="#">��Ʈ�Ͻ�</a></li>
                            </ul>
                        </div>
                    </li>
                    <li><a href="javascript:#void">����&�Ź�</a>
                        <div class="dropdown_nav">
                            <ul>
                                <li><a href="#">�Ż�ǰ</a></li>
                                <li><a href="#">��ü ��ǰ</a></li>
                                <li><a href="#">���� ����</a></li>
                                <li><a href="#">���� ����</a></li>
                                <li><a href="#">���� ����</a></li>
                                <li><a href="#">���� ����</a></li>
                            </ul>
                        </div>
                    </li>
                </ul>
                <ul class="right_menu">
                    <li><a href="#void" class="highlight">AI��õ�ڵ�</a></li>
                    <li><a href="#void" class="highlight">�ʴ뿩</a></li>
                    <li><a href="#void">Q&A</a></li>
                    <li><a href="#void">��������</a></li>
                </ul>
            </div>
        </div>
    </header>

    <div class="review-container inner">
        <div class="sidebar">
            <ul>
                <li><a href="#">���� �ֹ�</a></li>
                <li><a href="#">���� �뿩</a></li>
                <li><a href="#">��ٱ���</a></li>
                <li><a href="#">���� ��</a></li>
                <ul>
                    <li><a href="#">���Ǳ�</a></li>
                    <li><a href="#">����</a></li>
                </ul>
                <li><a href="#">ȸ������</a></li>
                <ul>
                    <li><a href="#">���� ����</a></li>
                    <li><a href="#">ȸ�� Ż��</a></li>
                </ul>
            </ul>
        </div>
        <div class="review-content">
            <div class="review-header">
                <h2>���� ����</h2>
            </div>
            <div class="review-form">
                <label>
                    <span>��ǰ��/�ɼ�</span>
                    <input type="text" name="product_name" placeholder="��ǰ��/�ɼ��� �Է��ϼ���">
                </label>
                <label>
                    <span>�ۼ�����</span>
                    <input type="text" name="review_date" placeholder="2024.05.29" disabled>
                </label>
                <label>
                    <span>����</span>
                    <input type="file" name="image">
                </label>
                <img src="../images/tshirt.jpg" alt="Ƽ���� �̹���">
                <label>
                    <span>����</span>
                    <div class="rating">
                        <input type="radio" name="rating" value="1"> 1
                        <input type="radio" name="rating" value="2"> 2
                        <input type="radio" name="rating" value="3"> 3
                        <input type="radio" name="rating" value="4"> 4
                        <input type="radio" name="rating" value="5" checked> 5
                    </div>
                </label>
                <label>
                    <span>����</span>
                    <textarea name="content" placeholder="������ �Է��ϼ���">���밨�� �ʹ� ���� ���� �ִ� ������� ��� �԰� �ٴ� �� ���ƿ�! ������õ�մϴ�~~</textarea>
                </label>
                <button>����ϱ�</button>
            </div>
        </div>
    </div>
</body>

</html>
    